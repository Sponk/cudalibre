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
				case '(': result += "BrOpen"; break;
				case ')': result += "BrClose"; break;
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
	std::string getFullyMangledName(FunctionDecl* d, bool includeClassname = false)
	{
		// If we found a builtin
		if(d->hasAttr<AnnotateAttr>())
		{
			if(d->getAttr<AnnotateAttr>()->getAnnotation().str() == "builtin")
				return d->getNameAsString();
			else if(d->getAttr<AnnotateAttr>()->getAnnotation().str() == "builtinf")
			{
				// Cut off the 'f' at the end
				std::string result = d->getNameAsString();
				result.pop_back();
				return result;
			}
		}

		std::string result;
		if(includeClassname && d->isCXXClassMember())
			result = getFullyMangledName(static_cast<CXXMethodDecl*>(d)->getParent()) + "_";

		if(d->isOverloadedOperator())
			result += getNewOperatorName(d->getNameAsString());
		else
			result += d->getNameAsString();
		
		for(size_t i = 0; i < d->getNumParams(); i++)
			result += "_" + d->getParamDecl(i)->getType().getAsString();
		
		std::replace(result.begin(), result.end(), ' ', '_');
		std::replace(result.begin(), result.end(), '*', 'p');
		std::replace(result.begin(), result.end(), '&', 'r');

		return result;
	}
	
	std::string getFullyMangledName(CXXRecordDecl* d)
	{
		std::string result = d->getTypedefNameForAnonDecl() 
					? d->getTypedefNameForAnonDecl()->getNameAsString()
					: d->getNameAsString();
		if(isa<ClassTemplateSpecializationDecl>(d))
		{
			auto decl = static_cast<ClassTemplateSpecializationDecl*>(d);
			for(auto k : decl->getTemplateInstantiationArgs().asArray())
			{
				switch(k.getKind())
				{
					case TemplateArgument::ArgKind::Type:
						result += "_" + k.getAsType().getAsString();
						break;

					case TemplateArgument::ArgKind::Integral:
						result += "_" + k.getAsIntegral().toString(16);
						break;

					default: llvm::report_fatal_error("Unknown template argument kind!");
				}
			}
		
			std::replace(result.begin(), result.end(), ' ', '_');
			std::replace(result.begin(), result.end(), '*', 'p');
			std::replace(result.begin(), result.end(), '&', 'r');
		}

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

	bool VisitAnnotateAttr(AnnotateAttr* attr)
	{
		/*if(attr->getAnnotation().str() == "local")
		{
			std::cout << rewriter.getRewrittenText(attr->getRange()) << std::endl;
		}*/

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
			case clang::Stmt::DeclRefExprClass:
			{
				DeclRefExpr* expr = static_cast<DeclRefExpr*>(s);
				VarDecl* varDecl = static_cast<VarDecl*>(expr->getReferencedDeclOfCallee());
				if(isa<ReferenceType>(varDecl->getType()))
				{
					rewriter.InsertTextBefore(expr->getLocation(), "(*");
					rewriter.InsertTextAfter(expr->getLocation().getLocWithOffset(varDecl->getNameAsString().size()), ")");
				}
			}
			break;

			case clang::Stmt::DeclStmtClass:
			{
				DeclStmt* stmt = static_cast<DeclStmt*>(s);

				if(!stmt->isSingleDecl())
					return true;

				auto decl = stmt->getSingleDecl();

				if(isa<VarDecl>(decl))
				{
					const auto varDecl = static_cast<VarDecl*>(decl);
					const auto type = varDecl->getType();
					
					if(isa<TemplateSpecializationType>(type))
					{
						std::stringstream newline;
						newline << getFullyMangledName(type->getAsCXXRecordDecl())
								<< " " << varDecl->getNameAsString();

						rewriter.ReplaceText(varDecl->getSourceRange(), newline.str());
					}
					else if(isa<ReferenceType>(varDecl->getType()))
					{
						transformReferenceParameter(varDecl);
						rewriter.InsertTextBefore(varDecl->getInit()->getLocStart(), "&");
					}

					if(varDecl->hasAttr<AnnotateAttr>())
					{
						const AnnotateAttr* attr = varDecl->getAttr<AnnotateAttr>();
						if(attr->getAnnotation().str() == "local")
						{
							rewriter.InsertTextBefore(varDecl->getOuterLocStart(), "__local ");
							rewriter.RemoveText(attr->getLocation().getLocWithOffset(-15), 34);
						}
						else if(attr->getAnnotation().str() == "constant")
						{
							rewriter.InsertTextBefore(varDecl->getOuterLocStart(), "__constant ");
							rewriter.RemoveText(attr->getLocation().getLocWithOffset(-15), 37);
						}

					}
				}
			}
			break;
			
			case clang::Stmt::CXXOperatorCallExprClass:
			{
				clang::CXXOperatorCallExpr* call = static_cast<clang::CXXOperatorCallExpr*>(s);

				auto decl = call->getCalleeDecl();
				if(!decl->isImplicit()
					&& decl->getAsFunction()->isOverloadedOperator())
				{
					if(call->isInfixBinaryOp())
					{

						TraverseStmt(call->getArg(0));
						TraverseStmt(call->getArg(1));

						rewriter.ReplaceText(call->getCallee()->getSourceRange(), ",");

						std::string line;
						line = getFullyMangledName(decl->getAsFunction(), true) + "(";
						line += rewriter.getRewrittenText(call->getSourceRange()) + ")";
						rewriter.ReplaceText(call->getSourceRange(), line);

					}
					else if(call->isAssignmentOp())
					{
						call->dump();
					}
					else
					{
						const size_t rangesize = rewriter.getRangeSize(call->getSourceRange());
						rewriter.InsertTextAfter(call->getLocStart().getLocWithOffset(rangesize), ")");

						rewriter.InsertTextBefore(call->getLocStart(),
												  getFullyMangledName(decl->getAsFunction(), true) + "(&");

						rewriter.RemoveText(call->getCallee()->getSourceRange());
					}
				}
			}
			break;

			case clang::Stmt::CallExprClass:
			{
				clang::CallExpr* call = static_cast<clang::CallExpr*>(s);

				auto decl = call->getCalleeDecl();
				if(decl && !decl->isImplicit())
				{
					auto declFunc = decl->getAsFunction();
					auto range = SourceRange(call->getLocStart(),
								call->getLocStart().getLocWithOffset(declFunc->getNameAsString().size() - 1));

					// Handle __syncthreads etc.
					if(declFunc->getNameAsString() == "__syncthreads")
					{
						rewriter.ReplaceText(call->getSourceRange(), "barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE)");
						return true;
					}
					else if(declFunc->getNameAsString() == "__threadfence_block")
					{
						rewriter.ReplaceText(call->getSourceRange(), "mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE)");
						return true;
					}

					for(int i = 0; i < declFunc->getNumParams(); i++)
					{
						const auto param = declFunc->getParamDecl(i);
						if(isa<ReferenceType>(param->getType()))
						{
							rewriter.InsertTextBefore(call->getArg(i)->getLocStart(), "&");
						}
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
						&& (t->getType() == base || !base->isStructureOrClassType())
						&& t->isImplicit())
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
		for(const FieldDecl* p : r->fields())
		{
			if(isa<ConstantArrayType>(p->getType()))
			{
				const ConstantArrayType* type = static_cast<const ConstantArrayType*>(p->getType()->getAsArrayTypeUnsafe());
				struc << "\t" << type->getElementType().getAsString() << " " << p->getNameAsString()
					  << "[" << type->getSize().toString(10, false) << "];" << std::endl;
			}
			else
 				struc << "\t" << p->getType().getAsString() << " " << p->getNameAsString() << ";" << std::endl;
		}
	}
	
	void writeStructBody(std::stringstream& struc, std::stringstream& methods, CXXRecordDecl* r)
	{
		/// The epilog that undefs stuff that got define e.g. for templates
		std::stringstream epilog;
		const std::string classname = getFullyMangledName(r);
						
		struc << "/// BEGIN FIELDS CLASS " << classname << std::endl;
		writeStructFields(struc, r);
		for(auto b : r->bases())
			writeStructFields(struc, b.getType()->getAsCXXRecordDecl());
		struc << "/// END FIELDS CLASS " << classname << std::endl;

		methods << "/// BEGIN METHODS CLASS " << classname << std::endl;

		methods << "#define this self" << std::endl;
		epilog << "#undef this" << std::endl;

		// If we got a template, define some names
		if(isa<ClassTemplateSpecializationDecl>(r))
		{
			auto decl = static_cast<ClassTemplateSpecializationDecl*>(r);
			auto pattern = decl->getTemplateInstantiationPattern();

			if(pattern)
			{
				auto parentDecl = pattern->getDescribedTemplate();

				for(int i = 0; i < parentDecl->getTemplateParameters()->size(); i++)
				{
					const auto arg = decl->getTemplateArgs()[i];
					const auto parentParam = static_cast<TemplateTypeParmDecl*>(parentDecl->getTemplateParameters()->getParam(i));

					switch(arg.getKind())
					{
						case TemplateArgument::ArgKind::Type:
						{
							const auto param = arg.getAsType();
							methods << "#define " << parentParam->getNameAsString() << " " << param.getAsString() << std::endl;
						} break;

						case TemplateArgument::ArgKind::Integral:
						{
							const auto param = arg.getAsIntegral();
							const auto parentParam = static_cast<TemplateTypeParmDecl*>(parentDecl->getTemplateParameters()->getParam(i));
							methods << "#define " << parentParam->getNameAsString() << " (0x" << param.toString(16) << ")" << std::endl;
						} break;

						default: llvm::report_fatal_error("Unknown template argument kind!");
					}

					epilog << "#undef " << parentParam->getNameAsString() << std::endl;
				}
			}
		}
		
		for(auto m : r->methods())
		{
			// A template method has its body in the template declaration
			// and not in the instanciated method itself. Handle that.
			Stmt* body = nullptr;
			if(!m->hasBody() 
			  && m->isTemplateInstantiation())
			{
				auto decl = static_cast<ClassTemplateSpecializationDecl*>(m->getParent());
				body = m->getTemplateInstantiationPattern()->getBody();
			}
			else
				body = m->getBody();
			
			// Ignore implicit functions and mere declarations
			if(body == nullptr || m->isImplicit())
				continue;

			for(auto stmt : body->children())
			{
				TraverseStmt(stmt);
			}

			const QualType type = m->getReturnType();
			if(isa<TemplateSpecializationType>(type))
			{
				methods << "\t" << getFullyMangledName(type->getAsCXXRecordDecl());
			}
			else if(type->isAnyPointerType() && isa<TemplateSpecializationType>(type->getPointeeType()))
			{
				methods << "\t" << getFullyMangledName(type->getPointeeType()->getAsCXXRecordDecl()) << "* ";
			}
			else
			{
				methods << "\t" << type.getAsString();
			}

			methods << " " << classname << "_" << getFullyMangledName(m)
					<< "(" << classname << "* self";

			for(auto a : m->parameters())
			{
				if(isa<ReferenceType>(a->getType()))
				{
					std::string type = a->getType().getAsString();
					std::replace(type.begin(), type.end(), '&', '*');
					methods << ", " << type << " " << a->getNameAsString();
				}
				else
				{
					methods << ", " << a->getType().getAsString() << " " << a->getNameAsString();
				}
			}
			
			methods << ")\n\t";
			methods << rewriter.getRewrittenText(body->getSourceRange()) << std::endl;
		}
		
		methods << epilog.str();
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

		auto templ = r->getDescribedTemplate();
		if(templ != nullptr || isa<ClassTemplateSpecializationDecl>(r))
		{
			SourceLocation start;
			SourceLocation end;
			std::string name;

			if(templ)
			{
				start = templ->getLocStart();
				end = templ->getSourceRange().getEnd();
				name = templ->getNameAsString();
			}
			else
			{
				start = r->getOuterLocStart();
				end = r->getSourceRange().getEnd();
				name = r->getNameAsString();
			}

			rewriter.InsertTextBefore(start, "/// BEGIN TEMPLATE " + name + "\n#if 0\n");
			
			rewriter.InsertTextAfter(findNextChar(end, ';'),
						 "\n/// END TEMPLATE " + name + "\n#endif\n");
						
			//rewriter.RemoveText(templ->getSourceRange());
		}
		else if(!r->isCLike() && !isa<ClassTemplateSpecializationDecl>(r))
		{
			std::stringstream struc, methods;
			struc << std::endl << "typedef struct " << std::endl << "{" << std::endl;
			writeStructBody(struc, methods, r);
			struc << "} " << getFullyMangledName(r) << ";" << std::endl << std::endl;
			struc << methods.str() << std::endl;

			// std::cout << struc.str() << std::endl;
			rewriter.ReplaceText(SourceRange(r->getOuterLocStart(),
					     r->getLocEnd().getLocWithOffset(1)), 
					     struc.str());
		}
		
		return true;
	}

	bool VisitTemplateSpecializationType(TemplateSpecializationType* s)
	{
		auto decl = s->getAsCXXRecordDecl();
		if(!decl->hasDefinition()
			|| !isa<ClassTemplateSpecializationDecl>(decl))
			return true;

		const std::string name = getFullyMangledName(decl);
		
		std::stringstream struc, methods;
		struc << std::endl << "typedef struct " << std::endl << "{" << std::endl;
		writeStructBody(struc, methods, decl);
		struc << "} " << name << ";" << std::endl << std::endl;
		struc << methods.str() << std::endl;

		const auto loc = findNextChar(decl->getSourceRange().getEnd(), ';');
		rewriter.InsertTextAfter(loc, struc.str());

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

	void transformReferenceParameter(VarDecl* a)
	{
		std::string line = rewriter.getRewrittenText(a->getSourceRange());

		std::replace(line.begin(), line.end(), '&', '*');

		const SourceLocation startLoc = a->getOuterLocStart();
		rewriter.ReplaceText(a->getSourceRange(), line);
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

				for(auto a : f->parameters())
				{
					if(isa<ReferenceType>(a->getType()))
					{
						transformReferenceParameter(a);
					}
				}
			}
		}

		return true;
	}

	bool VisitCXXStaticCastExpr(CXXStaticCastExpr* c)
	{
		TraverseStmt(c->getSubExpr());
		rewriter.ReplaceText(SourceRange(c->getLocStart(), c->getAngleBrackets().getEnd()), "(" + c->getType().getAsString() + ")");
	}

	bool VisitCXXDynamicCastExpr(CXXDynamicCastExpr* c)
	{
		TraverseStmt(c->getSubExpr());
		rewriter.ReplaceText(SourceRange(c->getLocStart(), c->getAngleBrackets().getEnd()), "(" + c->getType().getAsString() + ")");
	}

	bool VisitCXXReinterpretCastExpr(CXXReinterpretCastExpr* c)
	{
		TraverseStmt(c->getSubExpr());
		rewriter.ReplaceText(SourceRange(c->getLocStart(), c->getAngleBrackets().getEnd()), "(" + c->getType().getAsString() + ")");
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

int transformCudaClang(const std::string &code, std::string& result, const std::string& stdinc,
					   bool printIntermediate, bool printCl)
{
	if(printIntermediate)
		std::cout << std::endl << code << std::endl;

	auto frontend = new CLFrontendAction(result);
	// Transform to CL
	int retval = 0;
	retval = !runToolOnCodeWithArgs(frontend, code,
						{"-fsyntax-only",
							"-w",
							"-D__CUDACC__",
							"-D__CUDA_ARCH__", /// Since we are compiling GPU code
							"-D__CUDA_LIBRE_TRANSLATION_PHASE__",
							"-I/usr/cudalibre/include",
#ifdef RTINC
							RTINC,
#endif
							"-include", "translation_defines.h",
							"-D__kernel=__attribute__((annotate(\"kernel\")))",
							"-D__local=__attribute__((annotate(\"local\")))",
						   "-xc++"});

	if(retval)
	{
		std::cerr << "CUDA translation failed!" << std::endl;
		return retval;
	}

	if(printCl)
		std::cout << std::endl << result << std::endl;

	// Check syntax of produced CL code
	// @todo Add switch for additional syntax check!
	retval = !runToolOnCodeWithArgs(new SyntaxOnlyAction, result,
					{"-Wno-implicit-function-declaration",
					 "-w",
					 "-xcl", 
					 "-cl-std=CL2.0",
					 "-Dcl_clang_storage_class_specifiers",
					 "-isystem", stdinc + "/libclc/generic/include", /// @attention Don't hardcode paths like this!
					 "-isystem", "/usr/include/",
					 "-I/usr/include/cudalibre",

#ifdef RTINC
					 RTINC,
#endif
					 "-include", "clc/clc.h",
					 "-D__host__=/**/"
					 //"-include", "cuda_vectors.h",
					}, "input.cl");
	
	if(retval)
		std::cerr << "OpenCL syntax check failed!" << std::endl;

	return retval;
}
