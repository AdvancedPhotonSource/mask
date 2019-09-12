#include "masks.h"
#include <time.h>

void
project(
    float* obj, 
    const int ox, 
    const int oy, 
    const int oz,
    const float* gridx, 
    const float* gridy, 
    const float* gridz,
    const float dsrc,
    const float ddet,
    const float* detgridx,
    const float* detgridy,
    const int dx,
    const int dy,
    const float* srcgridx,
    const float* srcgridy,
    float* proj,
    const int px, 
    const int py)
{
    int m, n, mm, nn, a, indproj;
    float srcx, srcy, srcz, detx, dety, detz;
    float t, temp1, temp2, temp3;
    int numcoord, numind;
    float _dx, _dy, _dz;
    int gx = ox + 1;
    int gy = oy + 1;
    int gz = oz + 1;
    float* tlen = calloc(gx+gy+gz, sizeof(float));
    float* coordx = calloc(gx+gy+gz, sizeof(float));
    float* coordy = calloc(gx+gy+gz, sizeof(float));
    float* coordz = calloc(gx+gy+gz, sizeof(float));
    float* dist = calloc(gx+gy+gz, sizeof(float));
    float* mx = calloc(gx+gy+gz, sizeof(float));
    float* my = calloc(gx+gy+gz, sizeof(float));
    float* mz = calloc(gx+gy+gz, sizeof(float));
    int* ind = calloc(gx+gy+gz, sizeof(int));
    int* ix = calloc(gx+gy+gz, sizeof(int));
    int* iy = calloc(gx+gy+gz, sizeof(int));
    int* iz = calloc(gx+gy+gz, sizeof(int));

    // Tilt and rotate
    srcz = -dsrc;
    detz = ddet;

    // for (mm = 0; mm < ox*oy*oz; mm++)
    // {
    //     printf ("obj[%d]=%f \n", mm, obj[mm]);
    // }

    for (mm = 0; mm < dy; mm++)
    {
        dety = detgridy[mm];
        srcy = srcgridy[mm];

        for (nn = 0; nn < dx; nn++)
        {
            detx = detgridx[nn];
            srcx = srcgridx[nn];

            // Calculate plane intersections
            numcoord = 0;
            for(m = 0; m < gx; m++)
            { 
                t = (gridx[m] - srcx) / (detx - srcx);
                temp1 = srcy + t * (dety - srcy);
                temp2 = srcz + t * (detz - srcz);
                if (temp1 >= gridy[0])
                {
                    if  (temp1 <= gridy[oy])
                    {
                        if (temp2 >= gridz[0])
                        {
                            if  (temp2 <= gridz[oz])
                            {
                                tlen[numcoord] = t;
                                coordx[numcoord] = gridx[m];
                                coordy[numcoord] = temp1;
                                coordz[numcoord] = temp2;
                                numcoord++;
                            }
                        }
                    }
                }
            }
            for(m = 0; m < gy; m++)
            {
                t = (gridy[m] - srcy) / (dety - srcy);
                temp1 = srcx + t * (detx - srcx);
                temp2 = srcz + t * (detz - srcz);
                if (temp1 >= gridx[0])
                {
                    if  (temp1 <= gridx[ox])
                    {
                        if (temp2 >= gridz[0])
                        {
                            if  (temp2 <= gridz[oz])
                            {
                                tlen[numcoord] = t;
                                coordx[numcoord] = temp1;
                                coordy[numcoord] = gridy[m];
                                coordz[numcoord] = temp2;
                                numcoord++;
                            }
                        }
                    }
                }
            }
            for(m = 0; m < gz; m++)
            {
                t = (gridz[m] - srcz) / (detz - srcz);
                temp1 = srcx + t * (detx - srcx);
                temp2 = srcy + t * (dety - srcy);
                if (temp1 >= gridx[0])
                {
                    if  (temp1 <= gridx[ox])
                    {
                        if (temp2 >= gridy[0])
                        {
                            if  (temp2 <= gridy[oy])
                            {
                                tlen[numcoord] = t;
                                coordx[numcoord] = temp1;
                                coordy[numcoord] = temp2;
                                coordz[numcoord] = gridz[m];
                                numcoord++;
                            }
                        }
                    }
                }
            }

            // Sort coordinates in ascending order according to tlen
            for (m = 0; m < numcoord; m++)
            {
                for (n = 0; n < numcoord; n++)
                {
                    if (tlen[n] > tlen[m])
                    {
                        temp1 = tlen[m];
                        tlen[m] = tlen[n];
                        tlen[n] = temp1;

                        temp2 = coordx[m];
                        coordx[m] = coordx[n];
                        coordx[n] = temp2;

                        temp3 = coordy[m];
                        coordy[m] = coordy[n];
                        coordy[n] = temp3;

                        temp3 = coordz[m];
                        coordz[m] = coordz[n];
                        coordz[n] = temp3;
                    }  
                }
            }   

            // Calculate ray path lengths in each voxel
            numind = numcoord-1;
            for (m = 0; m < numind; m++)
            {
                _dx = (coordx[m + 1] - coordx[m]) * (coordx[m + 1] - coordx[m]);
                _dy = (coordy[m + 1] - coordy[m]) * (coordy[m + 1] - coordy[m]);
                _dz = (coordz[m + 1] - coordz[m]) * (coordz[m + 1] - coordz[m]);
                dist[m] = sqrtf(_dx + _dy + _dz);
            }

            // Calculate middle points of ray segments in each voxel
            for (m = 0; m < numind; m++)
            {
                mx[m] = coordx[m] + 0.5 * (coordx[m + 1] - coordx[m]);
                my[m] = coordy[m] + 0.5 * (coordy[m + 1] - coordy[m]);
                mz[m] = coordz[m] + 0.5 * (coordz[m + 1] - coordz[m]);
            }

            // Calc object's indices of ray path segments
            for (m = 0; m < numind; m++)
            {
                a = 0;
                for (n = 1; n < gx; n++)
                {
                    if (mx[m] <= gridx[n])
                    {
                        ix[m] = a;
                        break;
                    }
                    a++;
                }
            }
            for (m = 0; m < numind; m++)
            {
                a = 0;
                for (n = 1; n < gy; n++)
                {
                    if (my[m] <= gridy[n])
                    {
                        iy[m] = a;
                        break;
                    }
                    a++;
                }
            }
            for (m = 0; m < numind; m++)
            {
                a = 0;
                for (n = 1; n < gz; n++)
                {
                    if (mz[m] <= gridz[n])
                    {
                        iz[m] = a;
                        break;
                    }
                    a++;
                }
            }

            for (m = 0; m < numind; m++)
            {
                ind[m] = iz[m] + oz * iy[m] + oy * oz * ix[m];
            }

            // Calculate ray sum
            indproj = mm + py * nn;
            for (m = 0; m < numind; m++)
            {
                proj[indproj] += obj[ind[m]] * dist[m];
            }
        }
    }

    free(tlen);
    free(coordx);
    free(coordy);
    free(coordz);
    free(dist);
    free(mx);
    free(my);
    free(mz);
    free(ind);
    free(ix);
    free(iy);
    free(iz);
}

