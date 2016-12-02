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
	ASTVisitor(Rewriter &R) : rewriter(R) {}

	bool VisitRecordDecl(RecordDecl* r)
	{
		for(FieldDecl* c : r->fields())
		{
			// @todo Can't handle pointers for now. Needs runtime workaround for OpenCL.
			if(c->getType()->isAnyPointerType())
			{
				llvm::report_fatal_error("CudaLibre: Cannot handle pointers in structures right now!\n");
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

				for(int i = 0; i < f->getNumParams(); i++)
				{
					auto param = f->getParamDecl(i);
					if(param->getType()->isAnyPointerType())
					{
						SSBefore.str("");
						SSBefore << "__global ";

						rewriter.InsertText(param->getSourceRange().getBegin(), SSBefore.str(), true, true);
					}
				}
			}
		}

		return true;
	}

private:
	Rewriter &rewriter;
};

class CUDAASTConsumer : public ASTConsumer
{
public:
	CUDAASTConsumer(Rewriter &R) : visitor(R) {}
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
public:
	CUDAFrontendAction(std::string& resultStr) : result(resultStr) {}
	void EndSourceFileAction() override
	{
		SourceManager &SM = TheRewriter.getSourceMgr();
		TheRewriter.getEditBuffer(SM.getMainFileID()).write(result);
	}

	std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
												   StringRef file) override
	{
		TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
		return llvm::make_unique<CUDAASTConsumer>(TheRewriter);
	}

private:
	Rewriter TheRewriter;
};

std::string transformCudaClang(const std::string &code)
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
	auto frontend = new CUDAFrontendAction(result);

	runToolOnCodeWithArgs(frontend, src, {"-fsyntax-only", "-D__CLANG_CUDALIBRE__"});

	return result;
}
