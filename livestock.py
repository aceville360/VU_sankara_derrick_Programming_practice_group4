class PoultryBatch:
    def __init__(self, batch_id, record_date, chickens_count, sold, mortality):

        self.batch_id = batch_id
        self.record_date = record_date
        self.chickens_count = int(chickens_count)
        self.sold = int(sold)
        self.mortality = int(mortality)

        # PRIVATE MEMBERS (Encapsulation)
        # Using 11 grams per bird as the standard daily requirement
        self.__feed_per_bird_grams = 11
        self.__feed_inventory_kg = 0.0

    # GETTER METHOD: Safely reads the hidden inventory
    def get_inventory(self):
        return self.__feed_inventory_kg

    # SETTER METHOD: The gatekeeper method that prevents bad data
    def add_feed(self, amount):
        if amount < 0:
            print(
                f"Transaction Failed: Cannot add negative feed to Batch {self.batch_id}!")
        else:
            self.__feed_inventory_kg += amount
            print(
                f"Transaction Successful: Added {amount}kg. New stock: {self.__feed_inventory_kg}kg")

    # A method that uses the hidden 11g metric to calculate total feed
    def calculate_daily_feed(self):
        return (self.chickens_count * self.__feed_per_bird_grams) / 1000
