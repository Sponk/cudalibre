#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;
extern stringstream cppstream, clstream;

int parse(FILE *fp); // From parser.y

void replacestr(std::string& str, const std::string& search, const std::string& replace)
{
	for(size_t idx = 0;; idx += replace.length())
	{
		idx = str.find(search, idx);
		if(idx == string::npos)
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

int main(int argc, char **argv)
{
	// cout << "LibreCUDA compiler v0.1" << endl;

	if(argc < 3)
		return 0;

	FILE* f = fopen(argv[1], "r");
	if(!f)
	{
		perror("Could not open input file!");
		return 1;
	}

	int result = parse(f);
	ofstream cppout(argv[2]);

	if(!cppout)
	{
		perror("Could not open output file!");
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