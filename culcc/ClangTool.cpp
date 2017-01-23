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
#include <regex>

#include "GNUBlacklist.h"

#define VERSION_STRING "0.1"

#define PRINT_FILE(f) { std::cout << rewriter.getSourceMgr().getFilename((f)->getLocation()).str() << std::endl; }

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
	PrintIntermediate("print-intermediate", llvm::cl::desc("Print untranslated device code to stdout"), llvm::cl::Optional);

static llvm::cl::opt<bool>
	PrintCl("print-cl", llvm::cl::desc("Print translated device code to stdout"), llvm::cl::Optional);

static llvm::cl::opt<bool>
	GencodeSPIR("gencode-spir", llvm::cl::desc("Generate OpenCL SPIR binaries"), llvm::cl::Optional);
	
int compileSpir(const std::string& src, std::vector<unsigned char>& program);
int transformCudaClang(const std::string &code, std::string& result, const std::string& stdinc, bool printIntermediate, bool printCl);

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

		// Ensure only classes that do not depend on STL or similar are included.
		const bool isInList = fullname.empty() || headerBlacklist[fullname];
		if(isa<CXXRecordDecl>(d) && !isInList)
		{
			CXXRecordDecl* rd = static_cast<CXXRecordDecl*>(d);
			if(rd->getDefinition() == nullptr)
				return isInList;

			for(auto& b : rd->bases())
			{
				if(b.getType()->getAsTagDecl() != nullptr && !b.getTypeSourceInfo()->getTypeLoc().isNull())
					if(isInBlacklist(b.getType()->getAsTagDecl()))
						return true;
			}
		}

		return isInList;
	}

	bool VisitTypedefDecl(TypedefDecl* t)
	{
		if(!isInBlacklist(t) && t->getDescribedTemplate() == nullptr)
		{
			clResult << rewriter.getRewrittenText(t->getSourceRange()) << ";" << std::endl;
		}
	}

	bool VisitCXXRecordDecl(CXXRecordDecl* r)
	{
		if(!r->getNameAsString().empty()
			&& !isInBlacklist(r))
		{
			if(!r->hasDefinition() || (r->hasDefinition() && !r->isPolymorphic()))
			{
				//clResult << declToString(r, true) << ";" << std::endl;
				auto temp = r->getDescribedTemplate();
				if (temp == nullptr)
					clResult << declToString(r, true) /*rewriter.getRewrittenText(r->getSourceRange())*/ << ";" << std::endl;
				else
					clResult << rewriter.getRewrittenText(temp->getSourceRange()) << ";" << std::endl;
			}
		}

		return true;
	}

	std::string stmtToString(Stmt* stmt)
	{
		std::string result;
		llvm::raw_string_ostream out(result);
		PrintingPolicy policy(rewriter.getLangOpts());

		stmt->printPretty(out, NULL, policy);
		out.flush();

		return result;
	}

	std::string declToString(Decl* d, bool ignoreAttrs = false)
	{
		std::string result;
		llvm::raw_string_ostream out(result);
		PrintingPolicy policy(rewriter.getLangOpts());

		auto attrs = d->getAttrs();

		d->dropAttrs();
		d->print(out, policy);
		d->setAttrs(attrs);

		out.flush();
		return result;
	}


	bool VisitCUDAKernelCallExpr(CUDAKernelCallExpr* e)
	{
		FunctionDecl* decl = static_cast<FunctionDecl*>(e->getCalleeDecl());
		CallExpr* config = e->getConfig();

		std::stringstream newDecl;
		newDecl << "call" << decl->getNameAsString() << "("
				<< stmtToString(config->getArg(0)) << ", "
				<< stmtToString(config->getArg(1)) << (e->getNumArgs() ? ", " : "");

		rewriter.ReplaceText(SourceRange(e->getLocStart(), config->getLocEnd().getLocWithOffset(3)), newDecl.str());
		return true;
	}

	SourceLocation findNextOccurance(const SourceLocation& start, const std::string& c, int offset = 1)
	{
		const size_t sz = c.size();
		auto loc = start;
		std::string curr = " ";
		for(; loc.isValid() && curr != c; loc = loc.getLocWithOffset(offset))
		{
			const SourceRange range(loc, loc.getLocWithOffset(sz));
			curr = rewriter.getRewrittenText(range);
			std::cerr << curr << std::endl;
		}
		return loc;
	}

	/*bool VisitVarDecl(VarDecl* d)
	{
		if(isInBlacklist(d))
			return true;

		if(d->hasAttr<CUDASharedAttr>())
		{

			//const SourceLocation loc = findNextOccurance(d->getOuterLocStart(), "__shared__", -1);
			//const SourceLocation loc = d->getLocStart();
			std::cerr << "STUFF: " << rewriter.getRewrittenText(d->getSourceRange()) << std::endl;
			//rewriter.ReplaceText(SourceRange(loc, loc.getLocWithOffset(10)), "__local");
			//d->dump();
		}
		return true;
	}*/

/*	bool VisitStmt(Stmt* stmt)
	{
		switch(stmt->getStmtClass())
		{
			case clang::Stmt::DeclStmtClass:
			{
				DeclStmt* d = static_cast<DeclStmt*>(stmt);
				if(!d->isSingleDecl())
					return true;

				auto decl = d->getSingleDecl();

				if(isa<VarDecl>(decl) && decl->hasAttr<CUDASharedAttr>())
				{
					const auto varDecl = static_cast<VarDecl*>(decl);
					const auto type = varDecl->getType();

					rewriter.ReplaceText(varDecl->getSourceRange(), "<ALSDKFJ>");
					varDecl->dump();
				}
			}
			break;
		}

		return true;
	}*/

	bool VisitFunctionDecl(FunctionDecl *f)
	{
		const bool isGlobal = f->hasAttr<CUDAGlobalAttr>();
		const bool isDevice = f->hasAttr<CUDADeviceAttr>();
		const bool isBlacklisted = isInBlacklist(f);

		if (!isBlacklisted && (isGlobal || isDevice))
		{
			// Definition only needs to be removed
			if(!f->hasBody() || f->isCXXClassMember())
			{
				if(!f->isCXXClassMember())
				{
					int offset = -1;
					SourceLocation location = f->getLocEnd();
					SourceLocation end = f->getLocEnd().getLocWithOffset(1);
					std::string specifier = (isGlobal) ? "__global__" : "__device__";

					while (
						rewriter.getRewrittenText(SourceRange(location.getLocWithOffset(offset), end)).find(specifier)
							!= 0)
					{
						offset--;
					}

					rewriter.RemoveText(SourceRange(
						location.getLocWithOffset(offset),
						end));
				}

				return true;
			}

			if(isGlobal)
			{
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
			}
			
			int offset = -1;
			auto bodyLoc = f->getLocation();
			auto bodyLocEnd = f->getBody()->getLocEnd();

			std::string specifier = (isGlobal) ? "__global__" : "__device__";

			while (rewriter.getRewrittenText(SourceRange(bodyLoc.getLocWithOffset(offset), bodyLocEnd))
				.find(specifier) != 0)
				offset--;

			// Get function definition
			//clResult << "#if 0" << std::endl << stmtToString(f->getBody()) << std::endl << "#endif" << std::endl;
			/*clResult << (isGlobal ? "__kernel" : "")
					 << stmtToString(f->getBody())//rewriter.getRewrittenText(SourceRange(bodyLoc.getLocWithOffset(offset + specifier.size()),
						//										  bodyLocEnd))
					 << std::endl;*/

			clResult << (isGlobal ? "__kernel " : "") << declToString(f, true);

			if(f->isInlineSpecified())
			{
				while (rewriter.getRewrittenText(SourceRange(bodyLoc.getLocWithOffset(offset), bodyLocEnd))
							.find("inline") != 0)
				offset--;
			}
				
			if(!f->hasAttr<CUDAHostAttr>())
			{
				rewriter.RemoveText(SourceRange(
					bodyLoc.getLocWithOffset(offset),
					bodyLocEnd));
			}
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
	std::string stdinc;

public:
	CUDAFrontendAction(std::stringstream& cppResult, std::stringstream& clResult, const std::string& stdinc) :
		result(resultString),
		cppResult(cppResult),
		stdinc(stdinc),
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

		/// @todo Find proper solution using Clang. Somehow using a macro prevent clang from knowing a proper source location.
		//std::string clStr = std::regex_replace(clResult.str(), std::regex("__shared__"), "__local");
		//clStr = std::regex_replace(clStr, std::regex("__constant__"), "__constant");

		//clResult.str(clStr);

		std::string clOutput;
		if(transformCudaClang(clResult.str(), clOutput, stdinc, PrintIntermediate.getValue(), PrintCl.getValue()))
		{
			std::cerr << "Error while translating CUDA code!" << std::endl;
			exit(-1);
		}

		clResult.str(clOutput);

		// Write some comment to make understanding the generated code easier
		cppResult << "#include <cudalibre.h>" << std::endl
			  << "#ifndef __CUDA_LIBRE_PRIORITY__" << std::endl
			  << "#define __CUDA_LIBRE_PRIORITY__ 10" << std::endl
			  << "#endif" << std::endl
			  << "// Save the CUDA -> OpenCL translated code into a string" << std::endl << std::endl
			  << "// Use an anonymous namespace to provide an constructor function that sets up the runtime environment." << std::endl
			  << "namespace { " << std::endl
			  << "class LibreCudaInitializer { " << std::endl;
		
		if(GencodeSPIR.getValue() && false)
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
			cppResult << "static constexpr const char* cudalibre_clcode = " 
				  << "\"/// OPENCL_CODE_START\\n\"" << std::endl 
				  << stringify(clOutput)
 				  << "\"/// OPENCL_CODE_END\\n\"" << std::endl 
				  << ";" << std::endl
				  << "public: LibreCudaInitializer() { cu::initCudaLibre(cudalibre_clcode, __CUDA_LIBRE_PRIORITY__); }" << std::endl;
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
	const std::string& stdinc;
public:

	CUDAFrontendActionFactory(std::stringstream& cppResult, std::stringstream& clResult, const std::string& stdinc) :
		cppResult(cppResult),
		clResult(clResult),
		stdinc(stdinc){}

	clang::FrontendAction *create() override { return new CUDAFrontendAction(cppResult, clResult, stdinc); }
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
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-D__host__=__attribute__((host))"));
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-D__shared__=__attribute__((annotate(\"local\")))"));

	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-D__constant__=__attribute__((annotate(\"constant\")))"));

	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-D__CUDALIBRE_CLANG__"));
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-D__CUDACC__"));

#ifdef STDINC
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(STDINC));
#endif

#ifdef RTINC
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(RTINC));
#endif

	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-I/usr/include/cudalibre"));
	
	std::stringstream llvmVersion;
	// Calcualate manually since LLVM_VERSION_STRING might include some "svn" postfix
	/// @todo Only works on UNIX!
	llvmVersion << "-I/usr/lib/clang/"
				<< LLVM_VERSION_MAJOR << "." << LLVM_VERSION_MINOR << "." << LLVM_VERSION_PATCH
				<< "/include/";
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(llvmVersion.str().c_str()));

	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-include"));
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("math.cuh"));

	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-include"));
	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("cuda_runtime.h"));

	tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-std=c++14"));

	std::stringstream cppResult;
	std::stringstream clResult;

	auto frontendFactory = std::unique_ptr<CUDAFrontendActionFactory>(new CUDAFrontendActionFactory(cppResult, clResult, llvmVersion.str()));
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
