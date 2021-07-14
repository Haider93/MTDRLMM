#include "../../include/utilities/csv.h"

#include <limits>
#include <iostream>
#include <string>
using namespace std;

CSV::CSV(string& file_path)
{
    openFile(file_path);
}

void CSV::openFile(string& file_path)
{
    closeFile();

    fs.open(file_path);
    if (not fs.is_open()) {
        std::cout << "[!] Failed to open file: " << file_path << std::endl;
        exit(1);
    }
}

void CSV::closeFile()
{
    if (fs.is_open())
        fs.close();
}

bool CSV::isOpen() { return fs.is_open(); }
bool CSV::hasData() { return !fs.eof(); }

void CSV::parseRow(string& line, vector<string>& cols)
{
    size_t pos = 0;
    while (true) {
        size_t next = line.find(",", pos);

        if (next == string::npos) {
            cols.push_back(line.substr(pos, string::npos));
            break;
        } else {
            cols.push_back(line.substr(pos, next - pos));

            pos = next+1;
        }
    }
}

long prev_pos;//holds previous position of file stream reader

void CSV::next(vector<string>& cols)
{
    string line;
	prev_pos = fs.tellg();
    getline(fs, line);
    parseRow(line, cols);
}

void CSV::peek(vector<string>& cols)
{
    string line;

    long pos = fs.tellg();
    getline(fs, line);
    parseRow(line, cols);

    fs.seekg(pos);
}

void CSV::skip(int n_lines)
{
    for (int n = 0; n < n_lines; n++) {
        fs.ignore(numeric_limits<streamsize>::max(), fs.widen('\n'));

        if (not hasData()) return;
    }
}

long CSV::tellPosition() {
	return fs.tellg();
}

long CSV::tellPrevPosition() {
	return prev_pos;
}

void CSV::setPosition(long p) {
	fs.seekg(p);
}
