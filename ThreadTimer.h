#pragma once

class ThreadTimer
{
public:
	double StartTime;
	double StopTime;

	ThreadTimer(bool start = false);

	double GetElapsedTime(void);
	void Start(void);
	void Stop(void);

};

inline ThreadTimer::ThreadTimer(bool start)
{
	ThreadTimer::StartTime.QuadPart = 0;
	ThreadTimer::StopTime.QuadPart = 0;

	if (start)
		Start();
}

inline void ThreadTimer::Start(void)
{
	ThreadTimer::QueryPerformanceCounter(&StartTime);
}

inline void ThreadTimer::Stop(void)
{
	ThreadTimer::QueryPerformanceCounter(&StopTime);
}

