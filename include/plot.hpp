#pragma once
#include "config.hpp"
#include "vector.hpp"
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>

using namespace std;

struct plot_data_params
{
    Vector x, y;
    string config;
};

void plot(vector<plot_data_params> data, const char *title, const char *x_label, const char *y_label)
{
    char command[2048];
    snprintf(command, sizeof(command), R"(
    gnuplot -p -e "
        set title '%s';
        set xlabel '%s';
        set ylabel '%s';
        plot )",
    title, x_label, y_label);

    for (int i = 0; i < data.size(); i++)
    {
        plot_data_params &cur = data[i];
        if (cur.x.size() != cur.y.size())
        {
            cerr << "Cannot plot lines with mismatched sizes: " << cur.x.size() << ", " << cur.y.size() << endl;
            exit(EXIT_FAILURE);
        }

        string filename = "plot_data_" + to_string(i) + ".txt";
        ofstream file(filename);

        for (int i = 0; i < cur.x.size(); i++)
        {
            file << cur.x[i] << " " << cur.y[i] << '\n';
        }
        file.close();

        if(i != data.size() - 1)
        {
            cur.config += ",";
        }
        else 
        {
            cur.config += ";";
        }

        int ptr = strlen(command);
        snprintf(
            command + ptr,
            sizeof(command) - ptr,
            "'%s' %s", filename.c_str(), cur.config.c_str()
        );
    }

    snprintf(command + strlen(command), sizeof(command) - strlen(command), "\"");

    printf("Executing command:\n %s\n", command);
    system(command);
}