#include <bits/stdc++.h>
using namespace std;
int main(){
    #pragma omp parallel for schedule(dynamic , 2)
        for(int i = 0 ; i < 100 ; i += 2){
            
        }
}