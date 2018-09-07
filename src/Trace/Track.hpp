#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "pin.H"

using namespace std;

struct BLOCK
{
	string disassemble;
	UINT64 iterations;
	vector<pair<ADDRINT, OPCODE> > body;
	vector<pair<string, double> > entropy;
	bool loop = false;
	int reason;
};

class Track {
private:
	ofstream EntFile;
	bool enabled;
	string report;
	vector<string> files;
	vector<vector<int> > sentence;
	int t_loops = 0;
	int t_it = 0;
	int count = 0;

public:

	Track(vector<string> path, string ent_file);

	void activate(int opts);

	void addEvent(string event, bool imm);
	
	void write();

	int trackENT(vector<pair<string, double> > mem);

	int createWord(BLOCK bbl);

	string hex_to_string(const string& input);

	void trackBBL();

	void buildReport(map<ADDRINT, BLOCK>, vector<ADDRINT> bbl_order);

	void buildFeature(string data_set, string label);
};

