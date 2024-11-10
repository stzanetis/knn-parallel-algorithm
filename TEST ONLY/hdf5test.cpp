#include <iostream>
#include <H5Cpp.h>

using namespace std;

int main() {
    H5::H5File file("test.h5", H5F_ACC_TRUNC);
    cout << "File created" << endl;
    return 0;
}
