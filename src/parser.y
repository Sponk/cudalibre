%{
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <sys/types.h>
#include <cuda_defines.h>

#ifndef WIN32
#include <unistd.h>
#endif

#ifndef CUDALIBRE_STANDARD_INCLUDES
#define CUDALIBRE_STANDARD_INCLUDES "./"
#endif

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
	ss << "cu::callKernel(\"" << tmp << "\", ";
	
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

		ss << ", cu::ArgumentList({";
		for(int i = 0; i < args.size(); i++)
			ss << "CU_KERNEL_ARG(" << args[i] << ((i == args.size() - 1) ? ")" : "), ");
			//ss << "{ sizeof(" << args[i] << "), " << args[i] << "}" << ((i == args.size() - 1) ? "" : ", ");

		//ss << "{0, nullptr}}";
		ss << "}));";
	}

	str = ss.str();
	return str;
}

string& generateKernelDef(string& str)
{
	stringstream ss;

	int index = str.find("(") + 1;
	int closingIndex = str.find(")", index);

	ss << str.substr(0, index);

	vector<string> args;
	string arg = str.substr(index, closingIndex - index);
	splitstr(arg, ',', args);

	for(auto& s : args)
	{
		if(s.find("*") != -1)
		{
			ss << "__global ";
		}

		if(s != args.back())
		{
			ss << s << ", ";
		}
		else
		{
			ss << s << ")";
		}
	}

	ss << str.substr(closingIndex + 1);
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
%token <sval> SHARED
%token SPACE
%token <sval> KERNEL_CALL
%token <sval> TYPEDEF
%token <sval> STRUCT
%token <sval> INCLUDE
%token <sval> COMMENT

%type <sval> word
%type <sval> wordlist
%type <sval> line
%type <sval> linelist
%type <sval> global_function
%type <sval> device_function
%type <sval> shared_variable
%type <sval> code_block
%type <sval> spacelist
%type <sval> structure

%%

parser:
		| SPACE { cout << "SPACE" << endl; }
		| parser file 
		;

file:
	global_function { /*cppstream << *$1;*/ clstream << "__kernel " << generateKernelDef(*$1) << endl; delete $1; }
	| device_function { clstream << *$1 << endl; delete $1; }
	| shared_variable { clstream << *$1 << endl; delete $1; }
	| STRUCT { clstream << *$1 << endl; cppstream << *$1 << endl; delete $1; }
	| TYPEDEF { clstream << *$1 << endl; cppstream << *$1 << endl; delete $1; }
	| INCLUDE { clstream << *$1 << endl; cppstream << *$1 << endl; delete $1; }
	| KERNEL_CALL { cppstream << generateKernelCall(*$1) << endl; delete $1; }
	| linelist { cppstream << *$1; delete $1; }
	| CURLY_OPEN linelist CURLY_CLOSE { cppstream << "{" << *$2 << "}"; delete $2; }
	| COMMENT { cppstream << *$1 << endl; delete $1; }
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
	| linelist CURLY_CLOSE { *$1 += "}"; $$ = $1; }

// global_function does this now
//global_variable:
//	GLOBAL SPACE line { $$ = $3; }
//	; 

global_function:
	GLOBAL line code_block 
	{
		*$1 += *$2 + *$3; 
		delete $3; 
		delete $2; 
		$$ = $1; 
	}
	;

device_function:
	DEVICE line code_block 
	{ 
		// Remove __device__ keyword. It is not needed in OpenCL
		int idx = $1->find("__device__");
		if(idx != string::npos)
			$1->erase(idx, 10);
		
		*$1 += *$2 + *$3;
		delete $3;
		delete $2;
		$$ = $1; 
	}

shared_variable:
	SHARED line { *$1 += *$2; delete $2; $$ = $1; }	;

structure:
	STRUCT line code_block { *$1 += *$2 + *$3; delete $3; delete $2; $$ = $1; }
;

%%

#define LEXER_IMPLEMENTED

#ifndef WIN32
string runPreprocessor(const string& src)
{
	int fdParentChild[2];
	int fdChildParent[2];

	pipe(fdParentChild);
	pipe(fdChildParent);

	pid_t pid = fork();
	if(pid == -1)
		return src;

	// Child
	if(!pid)
	{
		close(fdParentChild[1]);
		close(fdChildParent[0]);

		dup2(fdParentChild[0], STDIN_FILENO);
		dup2(fdChildParent[1], STDOUT_FILENO);
		execl("/usr/bin/cpp", "/usr/bin/cpp", "-P", "-x", "c++", "-nostdinc", "-nostdinc++", "-I", CUDALIBRE_STANDARD_INCLUDES, (char*) 0);
	}
	else // Parent
	{
		close(fdParentChild[0]);
		close(fdChildParent[1]);
	}

	FILE* out = fdopen(fdParentChild[1], "w");
	fputs(src.c_str(), out);
	fflush(out);
	fclose(out);
	close(fdParentChild[1]);


	FILE* product = fdopen(fdChildParent[0], "r");
	string result;

	while(!feof(product))
		result += fgetc(product);

	fclose(product);

	// Remove EOF character
	result.erase(result.end() - 1);

	return result;
}
#else
string runPreprocessor(const string& src)
{
	return src;
}
#endif

static bool parserError = false;
int parse()
{
	clstream << cuda_header << endl;
	parserError = false;
	do {
#ifdef LEXER_IMPLEMENTED
		yyparse();

		if(parserError)
			return 1;
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

	std::string clcode = clstream.str();
	clstream.str(runPreprocessor(clcode));

	// cout << "Produced C++: " << endl << cppstream.str() << endl << "OpenCL: " << endl << clstream.str() << endl;
	return 0;
}

int parse(FILE *fp)
{
	yyin = fp;
	return parse();
}

extern void scan_string(const char* str);
extern int yy_scan_string(const char *);
extern int yylex_destroy();

int parse(const char* src)
{
	currline = 1;
	parserError = false;
	
#ifndef WIN32
	yy_scan_string(src);
#endif

	yyparse();
	
	if(parserError)
	{
		cout << "Parser error!" << endl;
		return 1;
	}

	std::string clcode = clstream.str();
	clstream.str(runPreprocessor(clcode));
	return 0;
}

void yyerror(const char *s)
{
	std::cout << "Parser error: " << s << " at " << currline << std::endl;
	parserError = true;
	//std::exit(EXIT_FAILURE);
}


