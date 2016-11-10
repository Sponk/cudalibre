%option noyywrap

%{

#include <iostream>
#include <vector>
#include <string>

#include "parser.hh"

#define YY_DECL extern "C" int yylex()
#define YY_NO_UNPUT

#ifdef WIN32
#define YY_NO_UNISTD_H

#include <io.h> // _isatty
#define isatty _isatty
#endif

extern int currline;
using namespace std;

%}

%%

__global__[ \t].+ { yylval.sval = new string(yytext); return GLOBAL; }
__device__[ \t].+ { yylval.sval = new string(yytext); return DEVICE; }
	/* __shared__[ \t].+ { yylval.sval = new string(yytext); return SHARED; }*/

struct[ \t]+(.*|\n|\{)*\}; { yylval.sval = new string(yytext); return STRUCT; }

typedef[ \t]+((struct[ \t]+)?(.*|\n|\{)*\}(.+)?;) { yylval.sval = new string(yytext); return TYPEDEF; }
typedef[ \t].+ { yylval.sval = new string(yytext); return TYPEDEF; }

[a-zA-Z0-9_]+<<<[a-zA-Z0-9_\(\)\+\-\*\/]+,[\ ]*[a-zA-Z0-9_]+>>>\(.*\); { yylval.sval = new std::string(yytext); return KERNEL_CALL; }

#include[ \t].+\"\.+\" { yylval.sval = new std::string(yytext); return INCLUDE; } /* This will not include system headers. */
#define[ \t].+ { yylval.sval = new std::string(yytext); return INCLUDE; }

"{" return CURLY_OPEN;
"}" return CURLY_CLOSE;
\n { currline++; return NEWLINE; }

. { yylval.character = yytext[0]; return CHARACTER; }

%%
/*[ ] { return 0; }
  \t { return 0; }
*/

void scan_string(const char* str)
{
    yy_switch_to_buffer(yy_scan_string(str));
}
