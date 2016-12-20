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

class ASTVisitor : public RecursiveASTVisitor<ASTVisitor>
{
public:
	ASTVisitor(Rewriter &R, std::stringstream& cppResult) : rewriter(R), cppResult(cppResult) {}

	bool VisitStmt(Stmt* s)
	{
		// If access belongs to a structure with pointer,
		// translate it!
		switch(s->getStmtClass())
		{
			case clang::Stmt::MemberExprClass:
			{
				clang::MemberExpr* member = static_cast<clang::MemberExpr*>(s);
				if (member->getType()->isAnyPointerType())
				{
					rewriter.ReplaceText(s->getSourceRange(),
										 rewriter.getRewrittenText(member->getBase()->getSourceRange())
											 + "_"
											 + member->getMemberNameInfo().getAsString());
				}
			}
			break;
		}

		return true;
	}

	bool VisitDecl(Decl* d)
	{
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

	bool VisitRecordDecl(RecordDecl* r)
	{
		for(FieldDecl* c : r->fields())
		{
			// @todo Can't handle pointers for now. Needs runtime workaround for OpenCL.
			if(c->getType()->isAnyPointerType())
			{
				rewriter.InsertTextBefore(c->getOuterLocStart(),
										  std::string("// Pointers in structs need to be emulated\n") +
				"unsigned int filler" + c->getNameAsString() + "; // This keeps the struct the right size for OpenCL\n// ");
				//llvm::report_fatal_error("CudaLibre: Cannot handle pointers in structures right now!\n");
			}
		}
		return true;
	}

	bool VisitFunctionDecl(FunctionDecl *f)
	{
		if (f->hasBody())
		{
			// Add __kernel if needed
			std::stringstream SSBefore;
			AnnotateAttr* attribute = f->getAttr<AnnotateAttr>();

			if(attribute && attribute->getAnnotation().str() == "global")
			{
				rewriter.RemoveText(attribute->getLocation().getLocWithOffset(-15), 35);
				SSBefore << "__kernel";
				SourceLocation ST = f->getSourceRange().getBegin();
				rewriter.InsertText(ST, SSBefore.str(), true, true);

				// Construct C++ wrapper
				std::stringstream cppArglist;
				//cppArglist << "void " << f->getNameAsString() << "(dim3 grid, dim3 block, ";
				cppArglist << "#define call" << f->getNameAsString() << "(grid, block";

				std::stringstream cppBody;
				cppBody << "\tcu::callKernel(\"" << f->getNameAsString() << "\", ";
				cppBody << "grid, block, cu::ArgumentList({";

				for(int i = 0; i < f->getNumParams(); i++)
				{
					auto param = f->getParamDecl(i);

					cppArglist << ", " << param->getNameAsString();
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

								// Insert pointer
								cppBody << "CU_KERNEL_ARG(" << param->getNameAsString() << "." << c->getNameAsString();
								cppBody << "), "; // Unconditional ',' since there is always the struct after the pointer
								// << ((i < f->getNumParams() - 1) ? ", " : "");
							}
						}
					}
					// Pointers are inserted before
					cppBody << "CU_KERNEL_ARG(" << param->getNameAsString();
					cppBody << ")" << ((i < f->getNumParams() - 1) ? ", " : "");
				}

				cppBody << "}));";

				cppArglist << ")\\" << std::endl;
				cppResult << cppArglist.str() << "{\\\n" << cppBody.str() << "\\\n}\n";

			}
		}

		return true;
	}

private:
	Rewriter& rewriter;
	std::stringstream& cppResult;
};

class CUDAASTConsumer : public ASTConsumer
{
public:
	CUDAASTConsumer(Rewriter &R, std::stringstream& cppResult) : visitor(R, cppResult) {}
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
	ASTVisitor visitor;
};

class CUDAFrontendAction : public ASTFrontendAction
{
	llvm::raw_string_ostream result;
	std::stringstream cppResult;
	std::string& cppResultStr;

public:
	CUDAFrontendAction(std::string& resultStr, std::string& cppStr) : result(resultStr), cppResultStr(cppStr) {}
	void EndSourceFileAction() override
	{
		SourceManager &SM = TheRewriter.getSourceMgr();
		TheRewriter.getEditBuffer(SM.getMainFileID()).write(result);
		cppResultStr = cppResult.str();
	}

	std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
												   StringRef file) override
	{
		TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
		return llvm::make_unique<CUDAASTConsumer>(TheRewriter, cppResult);
	}

private:
	Rewriter TheRewriter;
};

std::pair<std::string, std::string> transformCudaClang(const std::string &code)
{
	std::string src;

	src = "// Ensures our compiler does not cough up at OpenCL builtins.\n"
		"#ifdef __CLANG_CUDALIBRE__\n"
		"extern int get_num_groups(int);\n"
		"extern int get_local_size(int);\n"
		"extern int get_group_id(int);\n"
		"extern int get_local_id(int);\n"
		"#endif\n";

	src += code;

	std::string result;
	std::string cppResult;
	auto frontend = new CUDAFrontendAction(result, cppResult);

	runToolOnCodeWithArgs(frontend, src, {"-fsyntax-only", "-D__CLANG_CUDALIBRE__"});

	return std::pair<std::string, std::string>(result, cppResult);
}
