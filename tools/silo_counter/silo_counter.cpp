//  Copyright (c) 2019 Dominic C Marcello
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <silo.h>

#include <cstdio>
#include <string>

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        printf("silo_counter <prefix> <num_files>\n");
        return -1;
    }
    int maxblockindex = 0;
    int maxblocks = 0;
    int count = std::stoi(argv[2]);
    char* prefix = argv[1];
    for (int i = 0; i < count; i++)
    {
        std::string const name =
            std::string(prefix) + "." + std::to_string(i) + ".silo";
        auto db = DBOpenReal(name.c_str(), DB_HDF5, DB_READ);
        if (db)
        {
            auto mesh = DBGetMultimesh(db, "quadmesh");
            if (maxblocks < mesh->nblocks)
            {
                maxblockindex = i;
                maxblocks = mesh->nblocks;
            }
            printf("%i has %i blocks current max is %i\n", i,
                int(mesh->nblocks), maxblocks);
            DBFreeMultimesh(mesh);
            DBClose(db);
        }
        else
        {
            printf("%i is missing\n", i);
        }
    }
    printf("max blocks = %i at %i\n", maxblocks, maxblockindex);
    return 0;
}
