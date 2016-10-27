%{
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

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
%token GLOBAL
%token DEVICE
%token SPACE

%type <sval> word
%type <sval> wordlist
%type <sval> line
%type <sval> linelist
%type <sval> global_function
%type <sval> device_function
%type <sval> code_block;

%%

parser:
		| SPACE { cout << "SPACE" << endl; }
		| parser file 
		;

file:
	global_function { cppstream << *$1; clstream << *$1 << endl; delete $1; }
	| device_function { clstream << *$1 << endl; delete $1; }
	| linelist { cppstream << *$1; delete $1; }
	| CURLY_OPEN linelist CURLY_CLOSE { cppstream << "{" << *$2 << "}"; delete $2; }
	;

word: CHARACTER { $$ = new string; *$$ += $1; } 
	| word CHARACTER  { *$1 += $2; $$ = $1; }
	| word CURLY_OPEN { *$1 += "{"; $$ = $1; }
	| word CURLY_CLOSE { *$1 += "}"; $$ = $1; }
	;

wordlist: word { $$ = $1; }
	| wordlist SPACE word { *$1 += " " + *$3; delete $3; $$ = $1; }
	;
	
line: 	wordlist NEWLINE { *$1 += "\n"; $$ = $1; }
	| SPACE wordlist NEWLINE { *$2 += "\n"; $$ = $2; }
	| NEWLINE { $$ = new string("\n"); }
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
	GLOBAL SPACE line code_block { *$3 += *$4; delete $4; $$ = $3; }
	| SPACE GLOBAL SPACE line code_block { *$4 += *$5; delete $5; $$ = $4; }
	;
	
device_function:
	DEVICE SPACE line code_block { *$3 += *$4; delete $4; $$ = $3; }
	| SPACE DEVICE SPACE line code_block { *$4 += *$5; delete $5; $$ = $4; }
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

	cout << "Produced C++: " << endl << cppstream.str() << endl << "OpenCL: " << endl << clstream.str() << endl;
	return 0;
}

void yyerror(const char *s)
{
	std::cout << "Parser error: " << s << " at " << currline << std::endl;
	//std::exit(EXIT_FAILURE);
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
	
	// FIXME: ERROR HANDLING!
	FILE* f = fopen(argv[1], "r");
	int result = parse(f);
	
	ofstream cppout(std::string(argv[1]) + ".cpp");
	cppout << "static const char* clcode = " << stringify(clstream.str()) << endl << cppstream.str();
	
	return result;
}

