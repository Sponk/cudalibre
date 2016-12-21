#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <getopt.h>
#include <cstring>

#define VERSION_STRING "0.1"

using namespace std;
extern stringstream cppstream, clstream;

int parse(FILE *fp, const char* file); // From parser.y

void replacestr(std::string& str, const std::string& search, const std::string& replace);

std::string stringify(const std::string& str)
{
	std::string result = str;

	replacestr(result, "\\", "\\\\");
	replacestr(result, "\"", "\\\"");
	replacestr(result, "\n", "\\n\"\n\"");

	return "\"" + result + "\"";
}

void usage(const char* name)
{
	std::cout << name << " [-s input] [-o output]" << std::endl;
	std::cout << "\t-s:\tInput source file" << std::endl;
	std::cout << "\t-o:\tOutput C++ source file" << std::endl;
	std::cout << "\t-v:\tPrint version" << std::endl;
	std::cout << "\t-h:\tThis help" << std::endl;
}

struct Options
{
	std::string input;
	std::string output;
};

void parseArg(int argc, char** argv, Options& options)
{
	int opt;
	while((opt = getopt(argc, argv, "vhs:o:")) != -1)
	{
        switch (opt)
		{
        case 's':
			options.input = optarg;
            break;
			
        case 'o':
			options.output = optarg;
            break;

		case 'h':
			usage(argv[0]);
            exit(EXIT_SUCCESS);
			break;

		case 'v':
			std::cout << "clcc v" << VERSION_STRING << std::endl;
            exit(EXIT_SUCCESS);
			break;
			
        default:
			usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char **argv)
{
	// cout << "LibreCUDA compiler v0.1" << endl;

	/*if(argc < 3)
	{
		usage();
		return 0;
		}*/

	Options opt;
	parseArg(argc, argv, opt);
	
	FILE* f = fopen(opt.input.c_str(), "r");
	if(!f)
	{
		std::cerr << "Could not open \"" << opt.input << "\": " << strerror(errno) << std::endl;
		return 1;
	}

	int result = parse(f, opt.input.c_str());
	ofstream cppout(opt.output);

	if(!cppout)
	{
		std::cerr << "Could not open \"" << opt.output << "\": " << strerror(errno) << std::endl;
		return 1;
	}

	// Write some comment to make understanding the generated code easier
	cppout << "#include <cudalibre.h>" << endl;
	cppout << "// Save the CUDA -> OpenCL translated code into a string" << endl;
	cppout << "static const char* cudalibre_clcode = " << stringify(clstream.str()) << ";" <<  endl;
	cppout << endl << "// Use an anonymous namespace to provide an constructor function that sets up the runtime environment." << endl;
	cppout << "namespace { class LibreCudaInitializer { public: LibreCudaInitializer() { cu::initCudaLibre(cudalibre_clcode); } } init; }" << endl << endl;

	cppout << "// The C++ code written by the user" << endl;
	cppout << cppstream.str() << endl;

	return result;
}
