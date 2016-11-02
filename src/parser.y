%{
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cuda_defines.h>

extern "C" int yylex();
extern "C" int yyparse();
extern FILE *yyin;
extern char *yytext;
 
using namespace std;
 
void yyerror(const char *s);
void handle_command(char* cmd, std::vector<char*>* args);

#define YY_NO_UNPUT
#define YYERROR_VERBOSE

int currline = 1;
stringstream cppstream, clstream;

void replacestr(std::string& str, const std::string& search, const std::string& replace);

void splitstr(const std::string& str, char delim, std::vector<std::string>& elems)
{
	std::stringstream ss(str);

	std::string item;
	while (std::getline(ss, item, delim))
	{
		elems.push_back(item);
	}
}

string& generateKernelCall(string& str)
{
	stringstream ss;
	string tmp = str;
	tmp.erase(tmp.find("<<<"));
	ss << "lcCallKernel(\"" << tmp << "\", ";
	
	tmp = str;
	tmp = tmp.substr(tmp.find("<<<") + 3);
	tmp.erase(tmp.find(">>>"));
	ss << tmp;
	
	tmp = str;
	tmp = tmp.substr(tmp.find(">>>") + 4);
	tmp.erase(tmp.find_last_of(")"));
	
	if(tmp.empty())
		ss << ");";
	else
	{
		vector<string> args;
		splitstr(tmp, ',', args);

		ss << ", lcArgumentList({";
		for(int i = 0; i < args.size(); i++)
			ss << "LC_KERNEL_ARG(" << args[i] << ((i == args.size() - 1) ? ")" : "), ");
			//ss << "{ sizeof(" << args[i] << "), " << args[i] << "}" << ((i == args.size() - 1) ? "" : ", ");

		//ss << "{0, nullptr}}";
		ss << "}));";
	}

	str = ss.str();
	return str;
}

%}

%union{
  char		character;
  std::string*	sval;
}

%start parser 
%token <character> char_val
%token NEWLINE
%token <character> CHARACTER
%token CURLY_OPEN
%token CURLY_CLOSE
%token <sval> GLOBAL
%token <sval> DEVICE
%token SPACE
%token <sval> KERNEL_CALL

%type <sval> word
%type <sval> wordlist
%type <sval> line
%type <sval> linelist
%type <sval> global_function
%type <sval> device_function
%type <sval> code_block
%type <sval> spacelist

%%

parser:
		| SPACE { cout << "SPACE" << endl; }
		| parser file 
		;

file:
	global_function { /*cppstream << *$1;*/ clstream << "__kernel " << *$1 << endl; delete $1; }
	| device_function { clstream << *$1 << endl; delete $1; }
	| linelist { cppstream << *$1; delete $1; }
	| CURLY_OPEN linelist CURLY_CLOSE { cppstream << "{" << *$2 << "}"; delete $2; }
	;

word: CHARACTER { $$ = new string; *$$ += $1; } 
	| word CHARACTER  { *$1 += $2; $$ = $1; }
	| word CURLY_OPEN { *$1 += "{"; $$ = $1; }
	| word CURLY_CLOSE { *$1 += "}"; $$ = $1; }
	| word KERNEL_CALL { *$1 += generateKernelCall(*$2); $$ = $1; delete $2; }
	;

wordlist: word { $$ = $1; }
	| wordlist SPACE word { *$1 += " " + *$3; delete $3; $$ = $1; }
	;

spacelist: SPACE { $$ = new string(" "); }
	| spacelist SPACE { *$1 += " "; $$ = $1; }
	;

line: wordlist NEWLINE { *$1 += "\n"; $$ = $1; }
	| spacelist wordlist NEWLINE { *$1 += *$2 + "\n"; $$ = $1; delete $2; }
	| NEWLINE { $$ = new string("\n"); }
	//| line KERNEL_CALL { *$1 += generateKernelCall(*$2); $$ = $1; delete $2; }
	;

linelist: line { $$ = $1; }
	| linelist line { *$1 += " " + *$2; delete $2; $$ = $1; }
	;
	
code_block: { $$ = new string(); }
	| CURLY_OPEN linelist CURLY_CLOSE { *$2 = "{" + *$2 + "}"; $$ = $2; }

// global_function does this now
//global_variable:
//	GLOBAL SPACE line { $$ = $3; }
//	; 

global_function:
	GLOBAL line code_block { *$1 += *$2 + *$3; delete $3; delete $2; $$ = $1; }
	;
	
device_function:
	DEVICE line code_block { *$1 += *$2 + *$3; delete $3; delete $2; $$ = $1; }
	;

%%

#define LEXER_IMPLEMENTED

int parse(FILE *fp)
{
	yyin = fp;

	do {
#ifdef LEXER_IMPLEMENTED
		yyparse();
#else
		int x;
		std::cout << "Resulting tokens: ";
		while (x = yylex())
		{
			std::cout << "<" << yytext << "> ";
		}
		std::cout << std::endl;
#endif

	} while(!feof(yyin));
#ifndef LEXER_IMPLEMENTED
	std::exit(EXIT_SUCCESS);
#endif

	// cout << "Produced C++: " << endl << cppstream.str() << endl << "OpenCL: " << endl << clstream.str() << endl;
	return 0;
}

void yyerror(const char *s)
{
	std::cout << "Parser error: " << s << " at " << currline << std::endl;
	std::exit(EXIT_FAILURE);
}

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
	cout << "LibreCUDA compiler v0.1" << endl;
	
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
	
	cppout << "#include <cudalibre.h>" << endl;

	// Write some comment to make understanding the generated code easier
	cppout << "// Save the CUDA -> OpenCL translated code into a string" << endl;
	cppout << "static const char* librecuda_clcode = " << stringify(cuda_header) << endl << stringify(clstream.str()) << ";" <<  endl;
	cppout << endl << "// Use an anonymous namespace to provide an constructor function that sets up the runtime environment." << endl;
	cppout << "namespace { class LibreCudaInitializer { public: LibreCudaInitializer() { lcSetSources(librecuda_clcode); } } init; }" << endl << endl;
	cppout << cppstream.str();
	
	return result;
}

