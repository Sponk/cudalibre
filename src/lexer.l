%option noyywrap

%{

#include <iostream>
#include <vector>
#include "parser.hh"

#define YY_DECL extern "C" int yylex()
#define YY_NO_UNPUT

extern int currline;

%}

%%

__global__ return GLOBAL;
__device__ return DEVICE;

[a-zA-Z]+<<<[0-9]+,\ [0-9]+>>>\(.*\); { yylval.sval = new std::string(yytext); return KERNEL_CALL; }

"{" return CURLY_OPEN;
"}" return CURLY_CLOSE;
\n { currline++; return NEWLINE; }

[ ] { return SPACE; }

. { yylval.character = yytext[0]; return CHARACTER; }

%%
