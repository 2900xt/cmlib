#include "plot/plot.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cstdlib>

void plot(std::vector<plot_data_params> data, const char *title, const char *x_label, const char *y_label)
{
    if(system("command -v gnuplot > /dev/null 2>&1") != 0)
    {
        std::cerr << "gnuplot not found, skipping plot '" << title << "'" << std::endl;
        return;
    }
    if(system("mkdir -p data/tmp") != 0)
    {
        std::cerr << "Could not create data/tmp, skipping plot '" << title << "'" << std::endl;
        return;
    }

    char command[2048];
    snprintf(command, sizeof(command), R"(
    gnuplot -p -e "
        set title '%s';
        set xlabel '%s';
        set ylabel '%s';
        plot )",
    title, x_label, y_label);

    for (size_t i = 0; i < data.size(); i++)
    {
        plot_data_params &cur = data[i];
        if (cur.x.size() != cur.y.size())
        {
            std::cerr << "Cannot plot lines with mismatched sizes: " << cur.x.size() << ", " << cur.y.size() << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string filename = "data/tmp/plot_data_" + std::to_string(i) + ".txt";
        std::ofstream file(filename);

        for (size_t j = 0; j < cur.x.size(); j++)
        {
            file << cur.x[j] << " " << cur.y[j] << '\n';
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
    if(system(command) != 0)
    {
        std::cerr << "gnuplot failed" << std::endl;
    }
} 