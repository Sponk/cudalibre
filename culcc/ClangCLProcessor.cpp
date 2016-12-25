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
public:
	CLASTVisitor(Rewriter &R)
		: rewriter(R) {}

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

int transformCudaClang(const std::string &code, std::string& result)
{
	std::string src;

	src = "// Ensures our compiler does not cough up at OpenCL builtins.\n"
		"#ifdef __CLANG_CUDALIBRE__\n"
		"extern int get_num_groups(int);\n"
		"extern int get_local_size(int);\n"
		"extern int get_group_id(int);\n"
		"extern int get_local_id(int);\n"
		"#define __kernel __attribute__((annotate(\"kernel\")))\n"
		"#define __local __attribute__((annotate(\"local\")))\n"
		"struct dim3{int x; int y; int z;};\n"
		"dim3 threadIdx;\n"
		"#endif\n";

	src += "\n\n" + code;

	auto frontend = new CLFrontendAction(result);

	// Transform to CL
	int retval = 0;
	retval = !runToolOnCodeWithArgs(frontend, src,
						  {"-fsyntax-only",
						   "-D__CLANG_CUDALIBRE__",
						   "-xc++"});

	if(retval)
	{
		std::cerr << "CUDA translation failed!" << std::endl;
		//return retval;
	}

	// Check syntax of produced CL code
	// @todo Add switch for additional syntax check!
	retval = !runToolOnCodeWithArgs(new SyntaxOnlyAction, result,
									{"-Wno-implicit-function-declaration", "-xcl", "-cl-std=CL2.0"}, "input.cl");
	if(retval)
		std::cerr << "OpenCL syntax check failed!" << std::endl;

	return retval;
}
