#include <cstring>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <regex>

/// Some default flags so gcc and culcc can find headers and libraries
const std::string g_defaultFlags =
"-I/usr/include -I/usr/include/cudalibre -lOpenCL -Wl,-rpath=. -lclruntime";

// Switch-case on strings
#define SWITCH(x) { auto value = (x); if(false) {
#define CASE(x) } else if(std::regex_match(value, std::regex(x))) {
#define DEFAULT  } else {
#define IGNORE(x) CASE(x){}
#define HCTIWS }}

void usage(const char* name)
{
	std::cout << name << " [options] <source>" << std::endl;
	std::cout << "\t" << std::endl;
}

bool isArgnameEq(const char* name, const char* str)
{
	std::regex e(name);
	return !strcmp(name, str) || (*str == '-' && !strcmp(name, str+1));
}

struct Arguments
{
	std::string output;
	std::string input;
	std::vector<std::string> objects;
	std::vector<std::string> sources;
	bool hasC;
	bool verbose;
};

void parseArgs(int argc, 
	       char** argv,
	       std::stringstream& culccArgs,
	       std::stringstream& gccArgs,
	       Arguments& args)
{
	args.hasC = false;
	args.verbose = false;
	
	for(size_t i = 1; i < argc; i++)
	{
		char* arg = argv[i];
		
		SWITCH(arg)
			IGNORE("-?-cuda")
			IGNORE("-?-gencode=.*")
			IGNORE("-X.*")
			IGNORE("-arch.*")
			CASE(".*\\.o")
			{
				std::cout << "ARG: " << arg << std::endl;
				args.objects.push_back(arg);
			}
			CASE(".*\\.cu")
			{
				std::cout << "ARG: " << arg << std::endl;
				args.sources.push_back(arg);
			}
			CASE("-isystem.*")
			{
				gccArgs << "-I" << argv[++i] << " ";
			}
			CASE("-c|-?-compile")
			{
				args.hasC = true;
			}
			CASE("-o")
			{
				if(++i < argc)
				{
					args.output = argv[i];
				}
				else
				{
					std::cerr << "-o expects an argument!" << std::endl;
					exit(1);
				}
			}
			CASE("-v|-?-verbose")
			{
				args.verbose = true;
			}
			DEFAULT 
			{
				gccArgs << arg << " ";
			}
		HCTIWS
	}
	
	//args.input = argv[argc - 1];
	
	gccArgs << g_defaultFlags;
	culccArgs << " -- " << gccArgs.str();
}

int main(int argc, char** argv)
{
	if (argc <= 1)
	{
		usage(argv[0]);
		return 0;
	}

	std::stringstream culccArgs, gccArgs;
	Arguments args;
	parseArgs(argc, argv, culccArgs, gccArgs, args);
	
	// produces something like "culcc <source> -o <out> -- <some args>"
	// Generate CL + C++ for every cu file
	for(const auto& file : args.sources)
	{
		std::string intermediateCpp = file + ".cpp";
		std::string command = "culcc " + file + " -o " + intermediateCpp + culccArgs.str();
		if(args.verbose)
			std::cout << command << std::endl;
		
		if (system(command.c_str()))
		{
			std::cerr << "Could not preprocess CUDA file!" << std::endl;
			return 1;
		}
	}
	
	// Add source file here so it does not interfere with the culccArgs earlier

	if (!args.output.empty())
		gccArgs << " -o " << args.output;

	gccArgs << " " << (args.hasC ? "-c " : "") << args.input << ".cpp";
	
	std::string command = "g++ " + gccArgs.str();
	if(args.verbose)
		std::cout << command << std::endl;
		
	int retval = system(command.c_str());
	// Clean up afterwards
	//retval |= remove(intermediateCpp.c_str());
	return retval;
}
