from livestock import PoultryBatch

print("--- Farm Management System ---")
loaded_batches = []


print("\nLoading data from poultry.csv...")
try:
    with open('poultry.csv', 'r') as file:
        # Read all lines and skip the first row (the header)
        lines = file.readlines()[1:]

        for line in lines:
            # .split() automatically handles the spaces/tabs in your file
            data = line.split()
            if len(data) >= 5:
                batch_id = data[0]
                date = data[1]
                chickens = data[2]
                sold = data[3]
                mortality = data[4]

                # Turns the row of data into an Object
                batch_obj = PoultryBatch(
                    batch_id, date, chickens, sold, mortality)
                loaded_batches.append(batch_obj)

    print(
        f"Successfully loaded {len(loaded_batches)} records from the dataset.\n")

except FileNotFoundError:
    print("Error: Could not find poultry.csv. Make sure it is in the same folder.")

# 2. SHOWING ENCAPSULATION ON THE DATA
if len(loaded_batches) > 0:
    print("--- Testing Encapsulation ---")

    # Grab the very first row of data (Batch A1)
    test_batch = loaded_batches[0]

    print(f"Batch {test_batch.batch_id} on {test_batch.record_date} has {test_batch.chickens_count} active birds.")

    feed_needed = test_batch.calculate_daily_feed()
    print(f"Daily feed required: {feed_needed} kg")

    print("\nAttempting to force a negative feed inventory...")
    # This will be rejected by the setter method
    test_batch.add_feed(-50)

    print("\nAttempting a valid feed delivery...")
    # This will be accepted
    test_batch.add_feed(25)

    print(
        f"\nFinal verified inventory via getter: {test_batch.get_inventory()} kg")
