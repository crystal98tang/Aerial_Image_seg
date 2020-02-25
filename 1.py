class Car():
    def __init__(self, make, modle, year):
        self.make=make
        self.modle=modle
        self.year=year
        self.odometer_reading=0
    def get_descriptive_name(self):
        long_name=str(self.year)+' '+self.make+' '+self.modle
        return long_name.title()
    def update_odometer(self, mileage):
        if mileage>=self.odometer_reading:
            self.odometer_reading=mileage
        else:
            print("You can't roll back an odometer!")
    def increment_odometer(self,miles):
        self.odometer_reading+=miles
    def read_odometer(self):
        print("This car has "+str(self.odometer_reading)+" miles on it.")

class Battery():
    def __init__(self, battery_size=70):
        self.__battery_size=battery_size
    def describe_battery(self):
        print("This car has a "+str(self.__battery_size)+"-kWh battery.")

class ElectricCar(Car):
    def __init__(self, make, modle, year, size):
        super(make, modle, year).__init__(size)
        self.battery = Battery()

my_tesla=ElectricCar('tesla','modle s',2016)
print(my_tesla.get_descriptive_name())
my_tesla.battery.describe_battery()
