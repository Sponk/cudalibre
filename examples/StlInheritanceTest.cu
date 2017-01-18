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

template<typename T, typename T2>
class TemplateClass
{
public:
	T t;
	T2 l; /* Line Comment */
	T method(T in)
	{
		T tmp = in;
		return t + in;
	}

	__device__ T* operator()()
	{
		return &t;
	}

	__device__ T* operator()(T l)
	{
		t += l;
		return &t;
	}

	TemplateClass<T, T2>* operator+=(const T* t2)
	{
		t += *t2;
		return this;
	}
};

template<typename T, int size>
class Array
{
public:
	T data[size];
};

template<class T>
struct OutsideTemplate;

template<>
struct OutsideTemplate<unsigned char>
{
	float method(unsigned char val)
	{
		return 0.0f;
	}
};

__device__ void tempTest()
{
	TemplateClass<int, float> templateClassI;
	TemplateClass<float, int> templateClassF;
	Array<float, 12> array;

	int* pointer = templateClassI();
}

int main()
{

}
