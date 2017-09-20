#include <stdexcept>

using namespace std;

double ThreadTimer::GetElapsedTime(void)
{
	double elapsed_time;
	static double performance_freq;
	static bool qpf_called = false;

	if (!qpf_called)
	{
		QueryPerformanceFrequency((double*)&performance_freq);
		qpf_called = true;
	}

	elapsed_time = (StopTime.QuadPart - StartTime.QuadPart)*1000000;
	return (double)elapsed_time / performance_freq.QuadPart;
}

