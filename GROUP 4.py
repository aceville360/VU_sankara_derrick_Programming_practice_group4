
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")

mary = Person("Mary", 20)
peter = Person("Peter", 17)

mary.introduce()
peter.introduce()
