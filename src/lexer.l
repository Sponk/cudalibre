%option noyywrap

%{

#include <iostream>
#include <vector>
#include <string>

#include "parser.hh"

#define YY_DECL extern "C" int yylex()
#define YY_NO_UNPUT

extern int currline;
using namespace std;

%}

%%

__global__[ \t].+ { yylval.sval = new string(yytext); return GLOBAL; }
__device__[ \t].+ { yylval.sval = new string(yytext); return DEVICE; }

[a-zA-Z]+<<<[0-9]+,\ [0-9]+>>>\(.*\); { yylval.sval = new std::string(yytext); return KERNEL_CALL; }

"{" return CURLY_OPEN;
"}" return CURLY_CLOSE;
\n { currline++; return NEWLINE; }

. { yylval.character = yytext[0]; return CHARACTER; }

%%
/*[ ] { return 0; }
  \t { return 0; }
*/