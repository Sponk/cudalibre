//#include <vector>

/*struct Ignored : public std::vector<int>
{

};

struct IgnoredToo : public Ignored
{

};*/


struct s
{
	int val;
};

class Available
{
public:
	float value;
	float* value2;
	int size;
	struct s ParentS;
	
	void method()
	{
		
	}
	
	__device__ int method2(int x)
	{
		return x + 1;
	}
};

class AvailableToo : public Available
{
public:
	float value3;
	float* lala;
	int sz;
	struct s S;
	
	__device__ void lalaMethod()
	{
		struct s LOL;
		LOL.val++;
		sz++;
		size--;
		S.val++;
		ParentS.val++;
	}
	
	__device__ int inc(int l)
	{
		return l++;
	}
};

class AvailableToo2 : public Available
{
	float value3;
	float* lala;
	int sz;
	
	struct s lalalala;
};

class Test
{
public:
	int this_is_a_test;
	
private:
	float this_also;
};

__device__ void f(AvailableToo t)
{
	t.lalaMethod();
	t.sz = t.inc(54);
}

int main()
{

}
