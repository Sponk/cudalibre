#include <sstream>
#include <iostream>
#include <iomanip>
#include <memory>

#include <clang/AST/AST.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <stdlib.h>
#include <fstream>
#include <getopt.h>
#include <cstring>

#include <cudalibre.h>

#include "GNUBlacklist.h"

#define VERSION_STRING "0.1"

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

static llvm::cl::OptionCategory MyToolCategory("culcc options");
static llvm::cl::extrahelp CommonHelp("CudaLibre CUDA preprocessor v" VERSION_STRING);

static llvm::cl::extrahelp MoreHelp("\nA CUDA preprocessor that consumes CUDA code and produces C++14 and OpenCL 2.x code.");
static llvm::cl::opt<std::string>
	OutputFilename("o", llvm::cl::desc("<output file>"), llvm::cl::Required);

//static llvm::cl::opt<bool>
//	GencodeCL("gencode=cl", llvm::cl::desc("Generate OpenCL source code"), llvm::cl::Optional);
	
static llvm::cl::opt<bool>
	GencodeSPIR("gencode-spir", llvm::cl::desc("Generate OpenCL SPIR binaries"), llvm::cl::Optional);
	
int compileSpir(const std::string& src, std::vector<unsigned char>& program);
int transformCudaClang(const std::string &code, std::string& result);

class CUDAASTVisitor : public RecursiveASTVisitor<CUDAASTVisitor>
{
public:
	CUDAASTVisitor(Rewriter &R, std::stringstream& cppResult, std::stringstream& clResult)
		: rewriter(R), cppResult(cppResult), clResult(clResult) {}

	bool isInBlacklist(Decl* d)
	{
		std::string fullname = d->getASTContext().getSourceManager().getFilename(d->getLocation()).str();
		int idx = fullname.find_last_of("/");

		if(idx != std::string::npos)
			fullname = fullname.substr(idx + 1);

		return fullname.empty() || headerBlacklist[fullname];
	}

	bool VisitTypedefDecl(TypedefDecl* t)
	{
		if(!isInBlacklist(t))
		{
			clResult << rewriter.getRewrittenText(t->getSourceRange()) << ";" << std::endl;
		}
	}

	bool VisitRecordDecl(RecordDecl* r)
	{
		if(!r->getNameAsString().empty() && !isInBlacklist(r))
		{
			clResult << rewriter.getRewrittenText(r->getSourceRange()) << ";" << std::endl;
		}
		return true;
	}

	bool VisitCUDAKernelCallExpr(CUDAKernelCallExpr* e)
	{
		FunctionDecl* decl = static_cast<FunctionDecl*>(e->getCalleeDecl());
		CallExpr* config = e->getConfig();

		std::stringstream newDecl;
		newDecl << "call" << decl->getNameAsString() << "("
				<< rewriter.getRewrittenText(config->getArg(0)->getSourceRange()) << ", "
				<< rewriter.getRewrittenText(config->getArg(1)->getSourceRange()) << (e->getNumArgs() ? ", " : "");

		rewriter.ReplaceText(SourceRange(e->getLocStart(), config->getLocEnd().getLocWithOffset(3)), newDecl.str());
		return true;
	}

	bool VisitFunctionDecl(FunctionDecl *f)
	{
		bool isGlobal = f->hasAttr<CUDAGlobalAttr>();
		bool isDevice = f->hasAttr<CUDADeviceAttr>();

		if (!isInBlacklist(f) && (isGlobal || isDevice))
		{
			// Definition only needs to be removed
			if(!f->hasBody())
			{
				int offset = -1;
				SourceLocation location = f->getLocation();
				SourceLocation end = f->getLocEnd().getLocWithOffset(1);
				std::string specifier = (isGlobal) ? "__global__" : "__device__";

				while (rewriter.getRewrittenText(SourceRange(location.getLocWithOffset(offset), end))
					.find(specifier) != 0)
					offset--;

				rewriter.RemoveText(SourceRange(
					location.getLocWithOffset(offset),
					end));

				return true;
			}

			std::stringstream SSBefore;

			// Construct C++ wrapper
			std::stringstream cppArglist;
			cppArglist << "#define call" << f->getNameAsString() << "(grid, block";

			std::stringstream cppBody;
			cppBody << "\tcu::callKernel(\"" << f->getNameAsString() << "\", ";
			cppBody << "grid, block, cu::ArgumentList({";

			for(int i = 0; i < f->getNumParams(); i++)
			{
				auto param = f->getParamDecl(i);

				cppArglist << ", " << param->getNameAsString();
				if(param->getType()->isRecordType())
				{
					for(FieldDecl* c : static_cast<RecordDecl*>(param->getType()->getAsTagDecl())->fields())
					{
						// Pointer need to be moved out of the struct
						if(c->getType()->isAnyPointerType())
						{
							// Insert pointer
							cppBody << "CU_KERNEL_ARG(" << param->getNameAsString() << "." << c->getNameAsString();
							cppBody << "), "; // Unconditional ',' since there is always the struct after the pointer
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

			int offset = -1;
			auto bodyLoc = f->getLocation();
			auto bodyLocEnd = f->getBody()->getLocEnd();

			std::string specifier = (isGlobal) ? "__global__" : "__device__";

			while (rewriter.getRewrittenText(SourceRange(bodyLoc.getLocWithOffset(offset), bodyLocEnd))
				.find(specifier) != 0)
				offset--;

			// Get function definition
			clResult << (isGlobal ? "__kernel" : "")
					 << rewriter.getRewrittenText(SourceRange(bodyLoc.getLocWithOffset(offset + specifier.size()),
																  bodyLocEnd))
					 << std::endl;

			rewriter.RemoveText(SourceRange(
				bodyLoc.getLocWithOffset(offset),
				bodyLocEnd));
		}

		return true;
	}

private:
	Rewriter& rewriter;
	std::stringstream& cppResult;
	std::stringstream& clResult;
};

class CUDAASTConsumer : public ASTConsumer
{
public:
	CUDAASTConsumer(Rewriter &R, std::stringstream& cppResult, std::stringstream& clResult) : visitor(R, cppResult, clResult) {}
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
	CUDAASTVisitor visitor;
};

class CUDAFrontendAction : public ASTFrontendAction
{
	std::string resultString;
	llvm::raw_string_ostream result;
	std::stringstream& cppResult;
	std::stringstream& clResult;

public:
	CUDAFrontendAction(std::stringstream& cppResult, std::stringstream& clResult) :
		result(resultString),
		cppResult(cppResult),
		clResult(clResult){}

	void replacestr(std::string& str, const std::string& search, const std::string& replace)
	{
		for(size_t idx = 0;; idx += replace.length())
		{
			idx = str.find(search, idx);
			if(idx == std::string::npos)
				break;

			str.erase(idx, search.length());
			str.insert(idx, replace);
		}
	}

	std::string stringify(const std::string& str)
	{
		std::string result = str;

		replacestr(result, "\\", "\\\\");
		replacestr(result, "\"", "\\\"");
		replacestr(result, "\n", "\\n\"\n\"");

		return "\"" + result + "\"";
	}
	
	std::string byteify(const unsigned char* bytes, size_t size)
	{
		const int width = 20;
		std::stringstream output;
		for(int i = 0; i < size - 1; i++)
		{
			output << "0x" << std::setfill('0') << std::setw(2) << std::hex << (uint) bytes[i] << ", ";
			if(i % width == 0)
				output << std::endl;
		}
		output << "0x" << std::setfill('0') << std::setw(2) << std::hex << (uint) bytes[size - 1] << std::endl;
		
		return output.str();
	}

	void EndSourceFileAction() override
	{
		SourceManager &SM = TheRewriter.getSourceMgr();
		TheRewriter.getEditBuffer(SM.getMainFileID()).write(result);

		result.flush();

		std::string clOutput;
		if(transformCudaClang(clResult.str(), clOutput))
		{
			std::cerr << "Error while translating CUDA code!" << std::endl;
			//exit(-1);
		}

		clResult.str(clOutput);

		// Write some comment to make understanding the generated code easier
		cppResult << "#include <cudalibre.h>" << std::endl
			  << "// Save the CUDA -> OpenCL translated code into a string" << std::endl << std::endl
			  << "// Use an anonymous namespace to provide an constructor function that sets up the runtime environment." << std::endl
			  << "namespace { " << std::endl
			  << "class LibreCudaInitializer { " << std::endl;
		
		if(GencodeSPIR.getValue())
		{
			cu::SPIRHeader header;
			std::vector<unsigned char> program;
			compileSpir(clOutput, program);
			
			header.magic = cu::SPIRBIN_MAGIC;
			header.size = program.size();
			
			cppResult << "public: LibreCudaInitializer() { cu::initCudaLibreSPIR((const unsigned char[]) {"
				  << byteify((const unsigned char*) &header, sizeof(header)) << ", " << byteify(program.data(), program.size()) << "}); }" << std::endl;
		}
		else
		{
			cppResult << "static constexpr const char* cudalibre_clcode = " << stringify(clOutput) << ";" << std::endl
				  << "public: LibreCudaInitializer() { cu::initCudaLibre(cudalibre_clcode); }" << std::endl;
		}
		
		cppResult << "} init; }" << std::endl << std::endl;
		cppResult << "// The C++ code written by the user" << std::endl;
		cppResult << resultString << std::endl;
	}

	std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
												   StringRef file) override
	{
		TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
		return llvm::make_unique<CUDAASTConsumer>(TheRewriter, cppResult, clResult);
	}

	std::string getCppResult() const { return cppResult.str(); }
	std::string getCLResult() const { return clResult.str(); }

private:
	Rewriter TheRewriter;
};

using namespace clang::tooling;
using namespace llvm;

class CUDAFrontendActionFactory : public FrontendActionFactory
{
	std::stringstream& cppResult;
	std::stringstream& clResult;
public:

	CUDAFrontendActionFactory(std::stringstream& cppResult, std::stringstream& clResult) :
		cppResult(cppResult),
		clResult(clResult){}

	clang::FrontendAction *create() override { return new CUDAFrontendAction(cppResult, clResult); }
};

int main(int argc, char** argv)
{
	CommonOptionsParser opts(argc, (const char**) argv, MyToolCategory);
	ClangTool tool(opts.getCompilations(), opts.getSourcePathList());

	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-fsyntax-only"));
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-nocudainc"));
	//tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-nocudalib"));
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("--cuda-host-only"));
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-D__global__=__attribute__((global))"));
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-D__device__=__attribute__((device))"));
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-D__CUDALIBRE_CLANG__"));
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-I/usr/include/cudalibre"));

	std::stringstream llvmVersion;
	// Calcualate manually since LLVM_VERSION_STRING might include some "svn" postfix
	/// @todo Only works on UNIX!
	llvmVersion << "-I/usr/lib/clang/"
				<< LLVM_VERSION_MAJOR << "." << LLVM_VERSION_MINOR << "." << LLVM_VERSION_PATCH
				<< "/include/";
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(llvmVersion.str().c_str()));

	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-include"));
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("cuda_types.h"));
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-std=c++14"));

	std::stringstream cppResult;
	std::stringstream clResult;

	auto frontendFactory = std::unique_ptr<CUDAFrontendActionFactory>(new CUDAFrontendActionFactory(cppResult, clResult));
	int result = tool.run(frontendFactory.get());

	if(result != 0)
	{
		std::cerr << "Error while finding CUDA code!" << std::endl;
		return result;
	}

	std::ofstream out(OutputFilename.getValue());
	if(!out)
	{
		std::cerr << "Could not write output file!" << std::endl;
		return 1;
	}

	out << cppResult.str();

	return result;
}
