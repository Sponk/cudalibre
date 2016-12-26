#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <sstream>

int main(int argc, char** argv)
{
	std::vector<std::string> gccArgs;
	std::string source;
	std::string outfile;

	if(argc <= 1)
	{
		return system("g++");
	}

	for(int i = 1; i < argc; i++)
	{
		if(strstr(argv[i], "-gencode")
			|| strstr(argv[i], "-X")
			|| strstr(argv[i], "-arch"))
			continue;

		if(!strcmp("-o", argv[i]) && i < argc - 1)
		{
			outfile = argv[++i];
			continue;
		}
		else if(!strcmp("-isystem", argv[i]))
		{
			gccArgs.push_back("-I");
			continue;
		}

		gccArgs.push_back(argv[i]);
	}

	std::stringstream args;
	if(gccArgs.empty())
	{
		return 1;
	}

	source = gccArgs.back();
	gccArgs.pop_back();

	for(auto& s : gccArgs)
		args << " " << s;

	args << " -I/usr/include -I/usr/include/cudalibre -lOpenCL -Wl,-rpath=. -lclruntime";

	std::string culccArgs = "culcc ";

	// produces something like "culcc <source> -o <out> -- <some args>"
	culccArgs += source + " -o " + source + ".cpp -- " + args.str();

	if(system(culccArgs.c_str()))
	{
		std::cerr << "Could not preprocess CUDA file!" << std::endl;
		return 1;
	}

	// Add source file here so it does not interfere with the culccArgs earlier

	if(!outfile.empty())
		args << " -o " << outfile;

	args << " " << source << ".cpp";
	return system(("g++ " + args.str()).c_str());
}