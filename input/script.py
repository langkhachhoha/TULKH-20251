"""
Script to generate VRPTW test cases by permuting original test cases
Generates 5 versions for each problem size by reordering customers
"""

import random
import os


def read_test_case(filepath):
    """Read a test case file"""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    n = int(lines[0])
    
    # Read customer data
    customers = []
    for i in range(1, n + 1):
        e, l, d = map(int, lines[i].split())
        customers.append({
            'earliest': e,
            'latest': l,
            'duration': d
        })
    
    # Read travel time matrix
    travel_matrix = []
    for i in range(n + 1, n + 2 + n):
        row = list(map(int, lines[i].split()))
        travel_matrix.append(row)
    
    return n, customers, travel_matrix


def permute_test_case(n, customers, travel_matrix, permutation):
    """
    Permute customers according to given permutation
    permutation is a list like [2, 4, 1, 3, 5] meaning:
    - new customer 1 = old customer 2
    - new customer 2 = old customer 4
    - etc.
    """
    # Permutation is 1-indexed, convert to 0-indexed
    perm_0indexed = [p - 1 for p in permutation]
    
    # Permute customers
    new_customers = [customers[i] for i in perm_0indexed]
    
    # Permute travel matrix
    # New matrix[i][j] should be old matrix[perm[i]][perm[j]]
    # But row 0 and column 0 (depot) need special handling
    new_matrix = [[0] * (n + 1) for _ in range(n + 1)]
    
    # Depot row and column (index 0)
    for i in range(n):
        # Depot to customer i (new) = Depot to customer perm[i] (old)
        new_matrix[0][i + 1] = travel_matrix[0][perm_0indexed[i] + 1]
        new_matrix[i + 1][0] = travel_matrix[perm_0indexed[i] + 1][0]
    
    # Customer to customer
    for i in range(n):
        for j in range(n):
            old_i = perm_0indexed[i] + 1
            old_j = perm_0indexed[j] + 1
            new_matrix[i + 1][j + 1] = travel_matrix[old_i][old_j]
    
    return new_customers, new_matrix


def save_test_case(n, customers, travel_matrix, output_file):
    """Save test case to file"""
    with open(output_file, 'w') as f:
        # Write number of customers
        f.write(f"{n}\n")
        
        # Write customer data
        for customer in customers:
            f.write(f"{customer['earliest']} {customer['latest']} {customer['duration']}\n")
        
        # Write travel time matrix
        for i in range(n + 1):
            row_str = ' '.join(map(str, travel_matrix[i]))
            f.write(f"{row_str}\n")
    
    print(f"Generated: {output_file}")


def generate_permutations(n, num_versions=5, seed_base=42):
    """Generate different permutations for a given size"""
    permutations = []
    
    for version in range(num_versions):
        random.seed(seed_base + version)
        perm = list(range(1, n + 1))
        random.shuffle(perm)
        permutations.append(perm)
    
    return permutations


def generate_versions_from_original(original_file, num_versions=5):
    """Generate multiple versions from an original test case"""
    # Read original test case
    n, customers, travel_matrix = read_test_case(original_file)
    
    # Get base filename (e.g., N5.txt -> N5)
    base_name = os.path.splitext(os.path.basename(original_file))[0]
    output_dir = os.path.dirname(original_file)
    
    # Extract size for seed
    size = int(base_name[1:])  # Remove 'N' prefix
    
    # Generate permutations
    permutations = generate_permutations(n, num_versions, seed_base=size * 1000)
    
    # Generate each version
    for version, perm in enumerate(permutations, 1):
        # Create output filename
        output_file = os.path.join(output_dir, f"{base_name}_v{version}.txt")
        
        # Permute test case
        new_customers, new_matrix = permute_test_case(n, customers, travel_matrix, perm)
        
        # Save to file
        save_test_case(n, new_customers, new_matrix, output_file)


def generate_all_test_cases(input_dir=None):
    """
    Generate all test case versions from original files
    """
    # Use current directory if not specified
    if input_dir is None:
        input_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Original test case files
    original_files = [
        'N5.txt', 'N10.txt', 'N100.txt', 'N200.txt', 'N300.txt',
        'N500.txt', 'N600.txt', 'N700.txt', 'N900.txt', 'N1000.txt'
    ]
    
    print("="*60)
    print("VRPTW Test Case Generator (Permutation-based)")
    print("="*60)
    
    total_generated = 0
    
    for original_file in original_files:
        filepath = os.path.join(input_dir, original_file)
        
        if not os.path.exists(filepath):
            print(f"\nWarning: {original_file} not found, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Generating versions from {original_file}")
        print(f"{'='*60}")
        
        try:
            generate_versions_from_original(filepath, num_versions=5)
            total_generated += 5
        except Exception as e:
            print(f"Error processing {original_file}: {e}")
    
    print(f"\n{'='*60}")
    print(f"All test cases generated successfully!")
    print(f"Output directory: {input_dir}")
    print(f"Total files generated: {total_generated}")
    print(f"{'='*60}")


def verify_permutation(original_file, version_file):
    """Verify that a version file is a valid permutation of the original"""
    print(f"\nVerifying {os.path.basename(version_file)}...")
    
    # Read both files
    n1, customers1, matrix1 = read_test_case(original_file)
    n2, customers2, matrix2 = read_test_case(version_file)
    
    if n1 != n2:
        print(f"  ✗ Different number of customers: {n1} vs {n2}")
        return False
    
    # Check that customers are a permutation
    sorted_c1 = sorted([(c['earliest'], c['latest'], c['duration']) for c in customers1])
    sorted_c2 = sorted([(c['earliest'], c['latest'], c['duration']) for c in customers2])
    
    if sorted_c1 != sorted_c2:
        print(f"  ✗ Customers are not a permutation")
        return False
    
    print(f"  ✓ Valid permutation with {n1} customers")
    return True


def main():
    """
    Main function
    """
    input_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate all test cases
    generate_all_test_cases(input_dir)
    
    # Verify a few samples
    print(f"\n{'='*60}")
    print("Verifying sample test cases...")
    print(f"{'='*60}")
    
    samples = [
        ('N5.txt', 'N5_v1.txt'),
        ('N10.txt', 'N10_v1.txt'),
        ('N100.txt', 'N100_v1.txt')
    ]
    
    for original, version in samples:
        original_path = os.path.join(input_dir, original)
        version_path = os.path.join(input_dir, version)
        
        if os.path.exists(original_path) and os.path.exists(version_path):
            verify_permutation(original_path, version_path)


if __name__ == "__main__":
    main()
