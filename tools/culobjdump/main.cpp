#include <iostream>
#include <getopt.h>
#include <string>
#include <cstring>
#include <fstream>

struct Arguments
{
	std::string input;
	bool printHelp = false;
};

void usage(const char* callname)
{
	std::cout << "CudaLibre Object Inspector" << std::endl << std::endl;
	std::cout << "Usage: " << callname << " [-d file] " << std::endl << std::endl;
	std::cout << "-d  Print OpenCL sources" << std::endl;
}

void parseArguments(Arguments& arg, int argc, char** argv)
{
	int opt;
	while((opt = getopt(argc, argv, "d:")) != -1) 
	{
		switch (opt)
		{
			case 'd':
				arg.input = optarg;
				break;
		default:
			usage(argv[0]);
			exit(EXIT_FAILURE);
		}
	}
}

void dumpOpenCL(std::istream& in, std::ostream& out)
{
	std::string line;
	char tmp[26];
	tmp[25] = 0;

	while(!in.eof())
	{
		while(!in.eof())
		{
			std::getline(in, line);
			if(line.find("/// OPENCL_CODE_START") != -1)
				break;
		}

		while(!in.eof())
		{
			std::getline(in, line);
			if(line == "/// OPENCL_CODE_END")
				break;

			out << line << std::endl;
		}
	}
}

int main(int argc, char** argv)
{
	Arguments arg;
	parseArguments(arg, argc, argv);

	if(argc <= 1 || arg.printHelp)
	{
		usage(argv[0]);
		return 0;
	}

	std::ifstream in(arg.input);
	if(!in)
	{
		std::cerr << "Could not open object: " << strerror(errno) << std::endl;
		return 1;
	}

	dumpOpenCL(in, std::cout);
	return 0;
}

