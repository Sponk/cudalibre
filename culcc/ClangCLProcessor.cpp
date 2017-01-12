#include <sstream>
#include <iostream>

#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

class CLASTVisitor : public RecursiveASTVisitor<CLASTVisitor>
{
	std::string getNewOperatorName(const std::string& str)
	{
		std::string result;
		for(const char c : str)
			switch(c)
			{
				case '+': result += "Plus"; break;
				case '-': result += "Minus"; break;
				case '*': result += "Star"; break;
				case '/': result += "Slash"; break;
				case '=': result += "Equals"; break;
				case '<': result += "Smaller"; break;
				case '>': result += "Greater"; break;
			}
		
		return result;
	}

	/**
	 * @brief Returns an uniquely mangled name that respects parameter types for overloading.
	 *
	 * It will also handle the usage of builtin functions and translate them to their
	 * respective OpenCL or libcudastd call.
	 *
	 * @param d The declaration to translate.
	 * @return A new name.
	 */
	std::string getFullyMangledName(FunctionDecl* d)
	{
		// If we found a builtin
		if(d->hasAttr<AnnotateAttr>()
			&& d->getAttr<AnnotateAttr>()->getAnnotation().str() == "builtin")
		{
			return d->getNameAsString();
		}

		std::string result;
		if(d->isOverloadedOperator())
			result = getNewOperatorName(d->getNameAsString());
		else
			result = d->getNameAsString();
		
		for(size_t i = 0; i < d->getNumParams(); i++)
			result += "_" + d->getParamDecl(i)->getType().getAsString();
		
		std::replace(result.begin(), result.end(), ' ', '_');
		std::replace(result.begin(), result.end(), '*', 'p');
		std::replace(result.begin(), result.end(), '&', 'r');

		return result;
	}
	
public:
	CLASTVisitor(Rewriter &R)
		: rewriter(R) {}

	bool VisitCXXFunctionalCastExpr(CXXFunctionalCastExpr* c)
	{
		/// @note Translates like this: int(num) -> ((int)(num))
		rewriter.InsertTextBefore(c->getLocStart(), "((");
		rewriter.InsertTextBefore(c->getLParenLoc(), ")");
		rewriter.InsertTextBefore(c->getLocEnd(), ")");
		return true;
	}

	CXXThisExpr* containsThisExpr(Stmt* member)
	{
		if(isa<CXXThisExpr>(member))
		{
			CXXThisExpr* expr = static_cast<CXXThisExpr*>(member);
			return expr;
		}
		for(auto c : member->children())
		{
			CXXThisExpr* expr;
			if((expr = containsThisExpr(c)))
				return expr;
		}	
		return nullptr;
	}
	
	bool VisitStmt(Stmt* s)
	{
		/// @todo Needs to be cleaned up after each file!
		/// @fixme UGLY!
		static std::unordered_map<Stmt*, bool> handledStatements;
		if(handledStatements[s])
			return true;
		
		handledStatements[s] = true;
		
		// If access belongs to a structure with pointer,
		// translate it!
		switch(s->getStmtClass())
		{
			case clang::Stmt::CXXOperatorCallExprClass:
			{
				clang::CXXOperatorCallExpr* call = static_cast<clang::CXXOperatorCallExpr*>(s);

				auto decl = call->getCalleeDecl();
				if(!decl->isImplicit() && decl->getAsFunction()->isOverloadedOperator())
				{
					rewriter.InsertTextBefore(call->getLocStart(),
								  getFullyMangledName(decl->getAsFunction()) + "(");
					
					rewriter.InsertTextAfter(call->getLocEnd().getLocWithOffset(1), ")");
					rewriter.ReplaceText(call->getCallee()->getSourceRange(), ",");
				}
				
			}
			break;

			case clang::Stmt::CallExprClass:
			{
				clang::CallExpr* call = static_cast<clang::CallExpr*>(s);


				auto decl = call->getCalleeDecl();//->getAsFunction();
				if(decl && !decl->isImplicit())
				{
					auto declFunc = decl->getAsFunction();
					auto range = SourceRange(call->getLocStart(),
								call->getLocStart().getLocWithOffset(declFunc->getNameAsString().size() - 1));

					if(range.isInvalid())
					{
						llvm::report_fatal_error("Range is invalid!");
					}

					if(rewriter.getRangeSize(range) < 40)
						rewriter.ReplaceText(range, getFullyMangledName(declFunc));
				}
			}
			break;

			case clang::Stmt::MemberExprClass:
			{
				clang::MemberExpr* member = static_cast<clang::MemberExpr*>(s);
				
				{
					auto base = member->getBase()->getType();
					
					CXXThisExpr* t;
					if(member->getMemberDecl()->isDefinedOutsideFunctionOrMethod() 
						&& (t = containsThisExpr(member->getBase())) 
						&& (t->getType() == base || !base->isStructureOrClassType()))
					{
						rewriter.InsertTextBefore(member->getLocStart(), "self->");
					}
				}
				
				if (member->getType()->isAnyPointerType())
				{
					rewriter.ReplaceText(s->getSourceRange(),
										 rewriter.getRewrittenText(member->getBase()->getSourceRange())
											 + "_"
											 + member->getMemberNameInfo().getAsString());
				}
				else
				{
					char num = 0;
					switch(member->getMemberNameInfo().getAsString()[0])
					{
						case 'x':
							num = '0';
							break;

						case 'y':
							num = '1';
							break;

						case 'z':
							num = '2';
							break;

						default:
							return true;
					}

					std::string object = rewriter.getRewrittenText(member->getBase()->getSourceRange());
					if(object == "threadIdx")
					{
						rewriter.ReplaceText(
							SourceRange(member->getBase()->getLocStart(),
										member->getLocEnd()),
										std::string("get_local_id(") + num + ")");
					}
					else if(object == "blockIdx")
					{
						rewriter.ReplaceText(
							SourceRange(member->getBase()->getLocStart(),
										member->getLocEnd()),
							std::string("get_group_id(") + num + ")");
					}
					else if(object == "blockDim")
					{
						rewriter.ReplaceText(
							SourceRange(member->getBase()->getLocStart(),
										member->getLocEnd()),
							std::string("get_local_size(") + num + ")");
					}
					else if(object == "gridDim")
					{
						rewriter.ReplaceText(
							SourceRange(member->getBase()->getLocStart(),
										member->getLocEnd()),
							std::string("get_num_groups(") + num + ")");
					}
				}
			}
				break;
		}

		return true;
	}

	bool VisitDecl(Decl* d)
	{
		// If declaration is forced to be on the device for compilation
		if(d->hasAttr<AnnotateAttr>() && d->getAttr<AnnotateAttr>()->getAnnotation().str() == "device")
		{
			rewriter.RemoveText(SourceRange(d->getLocStart(), d->getLocEnd().getLocWithOffset(9)));
			return true;
		}

		if(d->getKind() == Decl::Var
			&& d->isLexicallyWithinFunctionOrMethod())
		{
			VarDecl* var = static_cast<VarDecl*>(d);

			if(var == nullptr
				|| var->getType()->getAsTagDecl() == nullptr
				|| var->getType()->getAsTagDecl()->getKind() != Decl::CXXRecord)
				return true;

			RecordDecl* recdecl = static_cast<RecordDecl*>(var->getType()->getAsTagDecl());

			for(FieldDecl* c : recdecl->fields())
			{
				// Pointer need to be moved out of the struct
				if(c->getType()->isAnyPointerType())
				{
					rewriter.InsertTextBefore(var->getLocStart(),
											  c->getType()->getPointeeType().getAsString()
												  + "* "
												  + var->getNameAsString()
												  + "_"
												  + c->getNameAsString() + ";\n"
					);
				}
			}
		}
		return true;
	}

	SourceLocation findNextChar(const SourceLocation& start, const char c, int offset = 1)
	{
		auto loc = start;
		char curr = 0;
		// std::cout << "START " << r->getNameAsString() << std::endl;
		for(; loc.isValid() && curr != c; loc = loc.getLocWithOffset(offset))
		{
			const SourceRange range(loc, loc);
			
			const std::string str = rewriter.getRewrittenText(range);
			if(str.empty())
				break;
			
			// std::cout << "\"" << str << "\"" << std::endl;
			curr = str.back();
			// str;
		}
		//std::cout << "END" << std::endl;
		return loc;
	}
	
	void writeStructFields(std::stringstream& struc, CXXRecordDecl* r)
	{
		for(auto p : r->fields())
		{
			struc << "\t" << p->getType().getAsString() << " " << p->getNameAsString() << ";" << std::endl;
		}
	}
	
	void writeStructBody(std::stringstream& struc, std::stringstream& methods, CXXRecordDecl* r)
	{
		const std::string classname = r->getTypedefNameForAnonDecl() 
						? r->getTypedefNameForAnonDecl()->getNameAsString()
						: r->getNameAsString();
						
		struc << "/// BEGIN FIELDS CLASS " << classname << std::endl;
		writeStructFields(struc, r);
		for(auto b : r->bases())
			writeStructFields(struc, b.getType()->getAsCXXRecordDecl());
		struc << "/// END FIELDS CLASS " << classname << std::endl;

		methods << "/// BEGIN METHODS CLASS " << classname << std::endl;
		for(auto m : r->methods())
		{
			for(auto stmt : m->getBody()->children())
			{
				TraverseStmt(stmt);
			}
			
			//methods << "\t" << rewriter.getRewrittenText(m->getSourceRange()) << std::endl;
			methods << "\t" << m->getReturnType().getAsString() << " " << classname << "_" << getFullyMangledName(m)
				<< "(" << classname << "* self";
				
			for(auto a : m->parameters())
			{
				methods << ", " << a->getType().getAsString() << " " << a->getNameAsString();
			}
			
			methods << ")\n\t";
			methods << rewriter.getRewrittenText(m->getBody()->getSourceRange()) << std::endl;
			
			//rewriter.RemoveText(m->getSourceRange());
		}
		methods << "/// END METHODS CLASS " << classname << std::endl;
	}
	
	bool VisitCXXRecordDecl(CXXRecordDecl* r)
	{
		for(FieldDecl* c : r->fields())
		{
			// @todo Can't handle pointers for now. Needs runtime workaround for OpenCL.
			if(c->getType()->isAnyPointerType())
			{
				rewriter.InsertTextBefore(c->getOuterLocStart(),
							  std::string("// Pointers in structs need to be emulated\n") +
							  "\tunsigned int filler" + c->getNameAsString() + 
							  "; // This keeps the struct the right size for OpenCL\n\t// ");
			}
		}
		
		if(!r->isCLike())
		{
			std::stringstream struc, methods;
			struc << std::endl << "typedef struct " << std::endl << "{" << std::endl;
			writeStructBody(struc, methods, r);
			struc << "} " << r->getNameAsString() << ";" << std::endl << std::endl;
			struc << methods.str() << std::endl;

			// std::cout << struc.str() << std::endl;
			rewriter.ReplaceText(SourceRange(r->getOuterLocStart(),
					     r->getLocEnd().getLocWithOffset(1)), 
					     struc.str());
		}
		
		return true;
	}

	bool VisitCXXMemberCallExpr(CXXMemberCallExpr* e)
	{
		std::stringstream newCall;
		const std::string& recordName = e->getRecordDecl()->getNameAsString();
		const CXXRecordDecl* implicitDecl = static_cast<CXXRecordDecl*>(e->getImplicitObjectArgument()
								   ->getReferencedDeclOfCallee());
		const std::string& implicitName = implicitDecl->getNameAsString();
		
		newCall << recordName << "_" << getFullyMangledName(e->getMethodDecl())
			<< "(" << ((implicitDecl->getTypeForDecl()->isAnyPointerType()) ? "" : "&") 
			<< implicitName;
			
		for(auto a : e->arguments())
		{
			newCall << ", " << rewriter.getRewrittenText(a->getSourceRange());
		}
		
		newCall << ")";
				
		rewriter.ReplaceText(e->getSourceRange(), newCall.str());
		return true;
	}
	
	bool VisitFunctionDecl(FunctionDecl *f)
	{
		// Fix operator overloading
		if(f->isOverloadedOperator())
		{
			rewriter.ReplaceText(f->getNameInfo().getSourceRange(), getFullyMangledName(f));
		}
		
		if (f->hasBody())
		{
			// Add __kernel if needed
			std::stringstream SSBefore;
			AnnotateAttr* attribute = f->getAttr<AnnotateAttr>();

			if(attribute && attribute->getAnnotation().str() == "kernel")
			{
				for(int i = 0; i < f->getNumParams(); i++)
				{
					auto param = f->getParamDecl(i);

					if(param->getType()->isAnyPointerType())
					{
						SSBefore.str("");
						SSBefore << "__global ";

						rewriter.InsertText(param->getSourceRange().getBegin(), SSBefore.str(), true, true);
					}
					else if(param->getType()->isRecordType())
					{
						for(FieldDecl* c : static_cast<RecordDecl*>(param->getType()->getAsTagDecl())->fields())
						{
							// Pointer need to be moved out of the struct
							if(c->getType()->isAnyPointerType())
							{
								rewriter.InsertTextBefore(param->getLocStart(),
											  "__global "
											  + c->getType()->getPointeeType().getAsString()
											  + "* "
											  + param->getNameAsString()
											  + "_"
											  + c->getNameAsString() + ", "
								);
							}
						}
					}
				}
			}
			else // __device__ function
			{
				if(!f->isCXXClassMember()) // Class members are moved around and mangled
					rewriter.ReplaceText(f->getNameInfo().getSourceRange(), getFullyMangledName(f));
			}
		}

		return true;
	}

private:
	Rewriter& rewriter;
};

class CLASTConsumer : public ASTConsumer
{
public:
	CLASTConsumer(Rewriter &R) : visitor(R) {}
	bool HandleTopLevelDecl(DeclGroupRef DR) override
	{
		for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b)
		{
			if(b)
				visitor.TraverseDecl(*b);
		}
		return true;
	}

private:
	CLASTVisitor visitor;
};


class CLFrontendAction : public ASTFrontendAction
{
	llvm::raw_string_ostream result;

public:
	CLFrontendAction(std::string& clResult) : result(clResult) {}
	void EndSourceFileAction() override
	{
		SourceManager &SM = TheRewriter.getSourceMgr();
		TheRewriter.getEditBuffer(SM.getMainFileID()).write(result);

		result.flush();
	}

	std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
							StringRef file) override
	{
		TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
		return llvm::make_unique<CLASTConsumer>(TheRewriter);
	}

private:
	Rewriter TheRewriter;
};

int transformCudaClang(const std::string &code, std::string& result, const std::string& stdinc)
{
	auto frontend = new CLFrontendAction(result);
	// Transform to CL
	int retval = 0;
	retval = !runToolOnCodeWithArgs(frontend, code,
						{"-fsyntax-only",
						   "-D__CUDACC__",
						   "-D__CUDA_ARCH__", /// Since we are compiling GPU code
						   "-D__CUDA_LIBRE_TRANSLATION_PHASE__",
#ifdef STDINC
						   STDINC,
#endif
						   "-I/usr/include/cudalibre",
						   "-include", "math.cuh",
						   "-xc++"});

	if(retval)
	{
		std::cerr << "CUDA translation failed!" << std::endl;
		return retval;
	}

	// std::cout << std::endl << result << std::endl;

	// Check syntax of produced CL code
	// @todo Add switch for additional syntax check!
	retval = !runToolOnCodeWithArgs(new SyntaxOnlyAction, result,
					{"-Wno-implicit-function-declaration", 
					 "-xcl", 
					 "-cl-std=CL2.0",
					 "-Dcl_clang_storage_class_specifiers",
					 "-isystem", stdinc + "/libclc/generic/include", /// @attention Don't hardcode paths like this!
					 "-isystem", "/usr/include/",
					 "-I/usr/include/cudalibre",
#ifdef STDINC
					 STDINC,
#endif

#ifdef RTINC
					 RTINC,
#endif
					 "-include", "clc/clc.h",
					 //"-include", "cuda_vectors.h",
					}, "input.cl");
	
	if(retval)
		std::cerr << "OpenCL syntax check failed!" << std::endl;

	return retval;
}
