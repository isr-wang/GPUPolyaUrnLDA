#include <iostream>

class x
{
  int* y;

public:
  x(int z)
  {
    y = new int[5];
    y[0] = z;
  }

  x(const x& name)
  {
    //...
  }

  void doSomething()
  {
    std::cout << y[0] << std::endl;
  }

  ~x()
  {
    delete[] y;
    std::cout << "deleted" << std::endl;
  }
};

void f(int x)
{
  x += 5;
  std::cout << x << std::endl;
}

void g(const int& x)
{
  // x += 5;
  std::cout << x << std::endl;
}

void h(int* x)
{
  *x += 5;
  std::cout << *x << std::endl;
}

int main()
{
  f(10);
  int x = 10;

  g(x);
  h((int*) x);
  g(x);
}
