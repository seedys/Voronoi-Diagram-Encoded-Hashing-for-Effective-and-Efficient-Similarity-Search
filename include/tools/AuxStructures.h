#ifndef AUX_STRUCTURES_H
#define AUX_STRUCTURES_H

#include <vector>
#include <cstdint>
#include <cstring>

namespace IFlib{

    struct VisitedTable
    {
        std::vector<uint8_t> visited;
        uint8_t visno;

        explicit VisitedTable(int size) : visited(size), visno(1) {}

        /// set flag #no to true
        void set(int no)
        {
            visited[no] = visno;
        }

        /// get flag #no
        bool get(int no) const
        {
            return visited[no] == visno;
        }

        /// reset all flags to false
        void advance()
        {
            visno++;
            if (visno == 250)
            {
                // 250 rather than 255 because sometimes we use visno and visno+1
                memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
                visno = 1;
            }
        }
    };

}// namespace IFlib
#endif // !AUX_STRUCTURES_H