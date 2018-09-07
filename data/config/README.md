# Generation Specification

## Primitives
Generatable algorithms are specified [here](pool). Each line specifies the algorithm, the type, and string literal name / description - respectively seperated by commas. 

### Notes
* Comments, whitespace and blank lines are not prohibited.

### Markup Tree
For each algorithm, a small JSON markup [fabrication tree](fab/) should provide the following:
1. Import statements.
2. Encryption routines.

## Labels
At run-time the procedural [engine](../../src/generate.py) will read the pre-set [labels](labels) to distinguish which cryptographic primitive(s) will be employed. The order of this list dictates it's numerical value, so after training it should not be modified.

### Notes
* Comments, whitespace and blank lines are disallowed.
* Supports multiple labels, but asymmetric routines should always come first.

## Compiler Flags 
Each app variant is compiled with one of the predefined [flags](flags).

### Notes
* Comments and blank lines are allowed.
* Only supports GCC.
