#include <iostream>
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <inttypes.h>
#include <sys/stat.h>
#include "pin.H"
#include "Track.hpp"

using namespace std;

/*
CryptoKnight
@gregorydhill
*/

// ====================
// | Global Variables |
// ====================

ofstream TraceFile;
PIN_LOCK lock;
ADDRINT main_begin;
ADDRINT main_end;

static ADDRINT WriteAddr;
static INT32 WriteSize;
static bool tail = false;

static UINT64 lines = 0; 
static UINT64 file = 0;
static string buffer = "";
static vector<string> path;
static string ent_file = "tmp/";

static string data_set;
static string label;
static string options;

#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

static void createBuffer() {
	lines = 0;
	TraceFile.close();
	string buff = "tmp/trace-file.bin.";
	buff.append(SSTR(file));
	path.push_back(buff);
	buffer = buff;
	TraceFile.open(buff.c_str());
	file++;
}

// =================
// | PIN Arguments |
// =================

KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool",
    "o", "Trace.out", "specify output file name");

KNOB<string> KnobLabel(KNOB_MODE_WRITEONCE, "pintool",
    "l", "", "specify label for sample");

KNOB<string> KnobOctal(KNOB_MODE_WRITEONCE, "pintool",
    "v", "3", "specify verbosity");

INT32 Usage()
{
	cerr << "CryptoKnight - Invalid Usage" << endl;
	return -1;
}

// ===============================
// | Calculate Shannon's Entropy |
// ===============================

double log2(double number) 
{
	double power = 2;
	return log(const_cast<double const&>(number)) / log(const_cast<double const&>(power));
}

double getEntropy(uint64_t value) 
{
	if (value==0)
		return 0;
	ostringstream o;
	o << value;
	string str(o.str());

	map<char, int> frequencies;

	for (const auto& s : str) frequencies[s]++;

	int numlen = str.length();
	double entropy = 0 ;

	for (const auto& f : frequencies) 
	{	
		double freq = static_cast<double>(f.second) / numlen;
		entropy += freq * log2(freq);
	}
	entropy *= -1;
	return entropy;
}

// =====================
// | Analysis Routines |
// =====================

string string_to_hex(const string& input)
{
	static const char* const lut = "0123456789ABCDEF";
	size_t len = input.length();

	string output;
	output.reserve(2 * len);
	for (size_t i = 0; i < len; ++i)
	{
		const unsigned char c = input[i];
		output.push_back(lut[c >> 4]);
		output.push_back(lut[c & 15]);
	}
	return output;
}
 
bool analyseWrite(ADDRINT addr, INT32 size, bool arg)
{
	WriteAddr = addr;
	WriteSize = size;
	return arg;
}
 
static void analyseINS(ADDRINT ip, INS ins, bool write)
{
	PIN_GetLock(&lock, ip);

	lines++;
	if (lines>=1000000)
		createBuffer();

	bool head = false;
	if (tail) head = true;

	if (INS_IsBranchOrCall(ins)||INS_IsRet(ins))
		tail = true;
	else tail = false;

	string dis = string_to_hex(INS_Disassemble(ins));

	TraceFile << StringFromAddrint(ip) << " " << StringFromUint64(INS_Opcode(ins)) << " " << dis << " " << head << " " << tail << " ";

	double ent = 0;
	uint64_t value;
	if (write) 
	{
		PIN_SafeCopy(&value, (void *)WriteAddr, WriteSize);
		if (WriteSize>8 && value!=0)
			ent = getEntropy(value);
		TraceFile << string_to_hex(REG_StringShort(INS_RegW(ins,0))) << " " << ent << endl;
	} 
	else {
		TraceFile << "0 " << "0" << endl;
	}

	PIN_ReleaseLock(&lock);
}

void Instruction(INS ins, VOID *v)
{
	ADDRINT ip = INS_Address(ins);
	if (ip < main_begin)
		return;

	bool write = false;
	if (INS_IsMemoryWrite(ins))
	{
		write = true;
		INS_InsertPredicatedCall(
			ins, IPOINT_BEFORE, (AFUNPTR)analyseWrite,
			IARG_MEMORYWRITE_EA,
			IARG_MEMORYWRITE_SIZE,
			IARG_EXECUTING,
			IARG_END);
    	}

	if (INS_HasFallThrough(ins)) {
		INS_InsertCall(
			ins, IPOINT_BEFORE, (AFUNPTR)analyseINS,
			IARG_INST_PTR,
			IARG_PTR, ins,
			IARG_BOOL, write,
			IARG_END);
	}
}

void ImageLoad(IMG img, void *v)
{
	PIN_GetLock(&lock, 0);
	if(IMG_IsMainExecutable(img))
	{
		size_t found = IMG_Name(img).find_last_of("/");
		ent_file.append(IMG_Name(img).substr(found+1));
		ent_file.append(".txt");

		main_begin = IMG_LowAddress(img);
		main_end = IMG_HighAddress(img);
	}
	PIN_ReleaseLock(&lock);
}

void End(INT32 code, VOID *v)
{
	TraceFile.close();
	Track exe(path, ent_file);

	int opts = atoi(options.c_str());
	if (opts & 2) exe.activate(opts);

	exe.trackBBL();
	exe.buildFeature(data_set, label);
	exe.write();
}

// ======================================
// | Register Instrumentation Functions |
// ======================================

int main(int argc, char * argv[])
{
	PIN_InitSymbols();
	PIN_Init(argc,argv);

	mkdir("tmp", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	createBuffer();
	
	data_set = KnobOutputFile.Value();
	label = KnobLabel.Value();
	options = KnobOctal.Value();

	int opts = atoi(options.c_str());

	if (opts & 1) 
	{
		IMG_AddInstrumentFunction(ImageLoad, 0);
		INS_AddInstrumentFunction(Instruction, 0);
		PIN_AddFiniFunction(End, 0);
	}

	PIN_StartProgram();
    
	return 0;
}
