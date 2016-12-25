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

	if(argc <= 1)
	{
		return system("g++");
	}

	for(int i = 1; i < argc; i++)
	{
		if(strstr(argv[i], "-gencode")
			|| strstr(argv[i], "-X"))
			continue;

		gccArgs.push_back(argv[i]);
	}

	std::stringstream args;
	if(gccArgs.empty())
	{
		return 1;
	}

	source = gccArgs.back();
	gccArgs.pop_back();

	args << "g++ -I/usr/include/cudalibre -lOpenCL -Wl,-rpath=. -lclruntime";
	for(auto& s : gccArgs)
		args << " " << s;

	args << " " << source << ".cpp";

	std::string culccArgs = "culcc ";
	culccArgs += source;

	if(system(culccArgs.c_str()))
	{
		std::cerr << "Could not preprocess CUDA file!" << std::endl;
		return 1;
	}

	return system(args.str().c_str());
}