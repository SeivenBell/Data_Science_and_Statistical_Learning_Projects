# Function to swap slices based on label values
def swap_slices(categories, images):
    for i in range(4):  # Loop over each position in the tensor
        # Find tensors that have 1 at the i-th position
        ones_indices = [idx for idx, cat in enumerate(categories) if cat[i] == 1]

        # Find tensors that have 0 at the i-th position
        zeros_indices = [idx for idx, cat in enumerate(categories) if cat[i] == 0]

        # Randomly select half from the ones_indices
        ones_to_swap = random.sample(ones_indices, len(ones_indices) // 2)

        # Randomly select the same number from the zeros_indices
        zeros_to_swap = random.sample(zeros_indices, len(ones_to_swap))

        # Perform the swap
        for one, zero in zip(ones_to_swap, zeros_to_swap):
            images[one][0, i], images[zero][0, i] = (
                images[zero][0, i].clone(),
                images[one][0, i].clone(),
            )

            # Also swap the corresponding labels
            categories[one][i], categories[zero][i] = (
                categories[zero][i],
                categories[one][i],
            )


swap_slices(categories, images)

new_categories = []
new_images = []
remove_count = 10_000
for cat, img in zip(categories, images):
    if str(cat.tolist()) == "[[0.0], [0.0], [0.0], [0.0]]":
        if remove_count > 0:
            remove_count -= 1
            continue
    new_categories.append(cat)
    new_images.append(img)


Counter([str(tensor.tolist()) for tensor in new_categories])


images = torch.concat(new_images, dim=0)
labels = torch.stack(new_categories)


# Create a dictionary to hold both tensors
data_dict = {"labels": labels, "images": images}

# Save the dictionary as a .pt file
torch.save(data_dict, "labels_and_images.pt")
