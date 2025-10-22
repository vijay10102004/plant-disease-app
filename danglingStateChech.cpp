#include<bits/stdc++.h>
using namespace std;
int main()
{
    int* pointer ;
    pointer = new int(5);

    cout<<*pointer<<endl;
    cout<<pointer<<endl;
    delete pointer;
    cout<<*pointer<<endl;
    cout<<pointer<<endl;
}