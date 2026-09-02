#pragma once

#include <chrono>
#include <sstream>

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0 / IM1)
#define IMM1 (IM1 - 1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1 + IMM1 / NTAB)
#define EPS 1.2e-7
#define RNMX (1.0 - EPS)
// See numerial recipes in C, chapter 7
float ran2(long* idum) {
    int j;
    long k;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB];
    float temp;
    if (*idum <= 0) {
        // Initialize.
        if (-(*idum) < 1)
            *idum = 1;
        // Be sure to prevent idum = 0.
        else
            *idum = -(*idum);
        idum2 = (*idum);
        for (j = NTAB + 7; j >= 0; j--) {
            // Load the shuffle table (after 8 warm-ups).
            k = (*idum) / IQ1;
            *idum = IA1 * (*idum - k * IQ1) - k * IR1;
            if (*idum < 0)
                *idum += IM1;
            if (j < NTAB)
                iv[j] = *idum;
        }
        iy = iv[0];
    }
    k = (*idum) / IQ1;
    // Start here when not initializing.
    *idum = IA1 * (*idum - k * IQ1) - k * IR1;
    // Compute idum=(IA1*idum) % IM1 without
    if (*idum < 0)
        *idum += IM1;
    // overflows by Schrage’s method.
    k = idum2 / IQ2;
    idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;
    // Compute idum2=(IA2*idum) % IM2 likewise.
    if (idum2 < 0)
        idum2 += IM2;
    j = iy / NDIV;
    // Will be in the range 0..NTAB-1.
    iy = iv[j] - idum2;
    // Here idum is shuffled, idum and idum2 are
    iv[j] = *idum;
    // combined to generate output.
    if (iy < 1)
        iy += IMM1;
    if ((temp = AM * iy) > RNMX)
        return RNMX;  // Because users don't expect endpoint values.
    else
        return temp;
}

void setRan(long& seed) {  // find totals sec since 0:00:00 and makes a seed with it
    std::stringstream date;
    auto today = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    date << ctime(&today);
    std::string month, day_name;
    int day, hour, min, sec, total_sec;
    char ch1, ch2;  //: :
    date >> day_name >> month >> day >> hour >> ch1 >> min >> ch2 >> sec;
    total_sec = -(hour * 360 + min * 60 + sec);  // seed must be negative
    seed = total_sec;
}