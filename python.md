# Python Data Structure Conversion Guide

## Complete List of Data Structures Covered

### Built-in Python Types
- **list** - Mutable sequence
- **tuple** - Immutable sequence
- **set** - Unordered unique elements
- **frozenset** - Immutable set
- **dict** - Key-value mapping
- **str** - String (text sequence)
- **bytes** - Immutable byte sequence
- **bytearray** - Mutable byte sequence

### Standard Library Types
- **array.array** - Typed array from array module
- **collections.deque** - Double-ended queue
- **collections.Counter** - Counting hashable objects
- **collections.defaultdict** - Dictionary with default values
- **collections.OrderedDict** - Dictionary that remembers insertion order
- **collections.namedtuple** - Tuple with named fields

### NumPy Types
- **numpy.ndarray** - N-dimensional array
- **numpy.matrix** - 2D array (deprecated, use ndarray)

### Pandas Types
- **pandas.Series** - 1D labeled array
- **pandas.DataFrame** - 2D labeled data structure
- **pandas.Index** - Immutable sequence used for indexing

---

## Conversion Syntax Reference

### 1. LIST Conversions

```python
my_list = [1, 2, 3, 4, 5]

# To other structures
tuple(my_list)                    # → tuple
set(my_list)                      # → set
frozenset(my_list)                # → frozenset
dict(enumerate(my_list))          # → dict with indices as keys
array.array('i', my_list)         # → array.array
np.array(my_list)                 # → numpy array
pd.Series(my_list)                # → pandas Series
pd.DataFrame(my_list)             # → pandas DataFrame (single column)
deque(my_list)                    # → deque
Counter(my_list)                  # → Counter
''.join(map(str, my_list))        # → string
bytes(my_list)                    # → bytes (if values 0-255)
```

### 2. TUPLE Conversions

```python
my_tuple = (1, 2, 3, 4, 5)

# To other structures
list(my_tuple)                    # → list
set(my_tuple)                     # → set
frozenset(my_tuple)               # → frozenset
dict(enumerate(my_tuple))         # → dict
array.array('i', my_tuple)        # → array.array
np.array(my_tuple)                # → numpy array
pd.Series(my_tuple)               # → pandas Series
pd.DataFrame(my_tuple)            # → pandas DataFrame
```

### 3. SET Conversions

```python
my_set = {1, 2, 3, 4, 5}

# To other structures
list(my_set)                      # → list (unordered)
tuple(my_set)                     # → tuple (unordered)
frozenset(my_set)                 # → frozenset
array.array('i', my_set)          # → array.array
np.array(list(my_set))            # → numpy array
pd.Series(list(my_set))           # → pandas Series
pd.DataFrame(list(my_set))        # → pandas DataFrame
```

### 4. DICTIONARY Conversions

```python
my_dict = {'a': 1, 'b': 2, 'c': 3}

# To other structures
list(my_dict.keys())              # → list of keys
list(my_dict.values())            # → list of values
list(my_dict.items())             # → list of (key, value) tuples
tuple(my_dict.items())            # → tuple of items
set(my_dict.keys())               # → set of keys
pd.Series(my_dict)                # → pandas Series (keys as index)
pd.DataFrame.from_dict(my_dict, orient='index')  # → DataFrame

# Dict of lists to DataFrame
dict_of_lists = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
pd.DataFrame(dict_of_lists)       # → DataFrame with columns
```

### 5. ARRAY.ARRAY Conversions

```python
my_array = array.array('i', [1, 2, 3, 4, 5])

# To other structures
list(my_array)                    # → list
tuple(my_array)                   # → tuple
set(my_array)                     # → set
np.array(my_array)                # → numpy array
pd.Series(my_array)               # → pandas Series
my_array.tobytes()                # → bytes
my_array.tolist()                 # → list (alternative)
```

### 6. NUMPY ARRAY Conversions

```python
my_numpy = np.array([1, 2, 3, 4, 5])

# To other structures
my_numpy.tolist()                 # → list
tuple(my_numpy)                   # → tuple
set(my_numpy)                     # → set
array.array('i', my_numpy)        # → array.array
pd.Series(my_numpy)               # → pandas Series
pd.DataFrame(my_numpy)            # → pandas DataFrame
my_numpy.tobytes()                # → bytes
my_numpy.astype(str)              # → string array

# 2D numpy to DataFrame
numpy_2d = np.array([[1, 2], [3, 4]])
pd.DataFrame(numpy_2d, columns=['A', 'B'])  # → DataFrame with column names
```

### 7. PANDAS SERIES Conversions

```python
my_series = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])

# To other structures
my_series.tolist()                # → list
tuple(my_series)                  # → tuple
set(my_series)                    # → set
my_series.to_dict()               # → dict (index as keys)
array.array('i', my_series)       # → array.array
my_series.to_numpy()              # → numpy array
my_series.values                  # → numpy array (values only)
my_series.to_frame()              # → DataFrame (single column)
my_series.to_frame(name='col')    # → DataFrame with column name
```

### 8. PANDAS DATAFRAME Conversions

```python
my_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

# To other structures
my_df.values.tolist()             # → nested list
my_df.to_dict('records')          # → list of dicts (row-oriented)
my_df.to_dict('list')             # → dict of lists (column-oriented)
my_df.to_dict('series')           # → dict of Series
my_df.to_dict('dict')             # → nested dict
my_df.to_numpy()                  # → numpy array (2D)
my_df['col1']                     # → Series (single column)
my_df['col1'].tolist()            # → list (single column)
list(my_df.itertuples(index=False, name=None))  # → list of tuples
my_df.to_json()                   # → JSON string
my_df.to_csv()                    # → CSV string
my_df.to_records()                # → numpy record array
```

### 9. COLLECTIONS Module Conversions

```python
from collections import deque, Counter, defaultdict, OrderedDict

# DEQUE
my_deque = deque([1, 2, 3, 4, 5])
list(my_deque)                    # → list
tuple(my_deque)                   # → tuple
set(my_deque)                     # → set

# COUNTER
my_counter = Counter([1, 2, 2, 3, 3, 3])
list(my_counter.elements())       # → list with repetitions
dict(my_counter)                  # → dict
list(my_counter.items())          # → list of (element, count) tuples
my_counter.most_common()          # → list of (element, count) sorted
pd.Series(my_counter)             # → pandas Series

# ORDEREDDICT
my_ordered = OrderedDict([('a', 1), ('b', 2)])
dict(my_ordered)                  # → regular dict
list(my_ordered.items())          # → list of items
pd.Series(my_ordered)             # → pandas Series

# DEFAULTDICT
my_default = defaultdict(int, {'a': 1, 'b': 2})
dict(my_default)                  # → regular dict
```

### 10. STRING and BYTES Conversions

```python
# STRING conversions
my_string = "1,2,3,4,5"
my_string.split(',')              # → list (split by delimiter)
list(my_string)                   # → list of characters
tuple(my_string)                  # → tuple of characters
set(my_string)                    # → set of unique characters
my_string.encode('utf-8')         # → bytes

# BYTES conversions
my_bytes = b'hello'
list(my_bytes)                    # → list of byte values
bytearray(my_bytes)               # → bytearray
my_bytes.decode('utf-8')          # → string
np.frombuffer(my_bytes, dtype=np.uint8)  # → numpy array
```

---

## Special Conversion Patterns

### Nested Structures

```python
# Nested list to DataFrame
nested_list = [[1, 2, 3], [4, 5, 6]]
pd.DataFrame(nested_list, columns=['A', 'B', 'C'])

# List of dicts to DataFrame
list_of_dicts = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
pd.DataFrame(list_of_dicts)

# DataFrame to nested list
df.values.tolist()

# Dict of lists to DataFrame
pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
```

### Multi-Index and Complex Structures

```python
# MultiIndex Series to DataFrame
multi_series.reset_index()

# DataFrame with MultiIndex to dict
df.to_dict('index')

# Transpose DataFrame
df.T  # or df.transpose()
```

### Combining Lists/Arrays

```python
# Two lists to dict
dict(zip(keys_list, values_list))

# Multiple lists to DataFrame
pd.DataFrame(list(zip(list1, list2, list3)), columns=['A', 'B', 'C'])

# Concatenate arrays/Series
np.concatenate([arr1, arr2])
pd.concat([series1, series2])
```

---

## Quick Reference Matrix

| FROM ↓ / TO → | list | tuple | set | dict | array | numpy | Series | DataFrame |
|---------------|------|-------|-----|------|-------|-------|--------|-----------|
| **list** | — | `tuple()` | `set()` | `enumerate` | `array()` | `np.array()` | `pd.Series()` | `pd.DataFrame()` |
| **tuple** | `list()` | — | `set()` | `enumerate` | `array()` | `np.array()` | `pd.Series()` | `pd.DataFrame()` |
| **set** | `list()` | `tuple()` | — | N/A | `array(list)` | `np.array()` | `pd.Series()` | `pd.DataFrame()` |
| **dict** | `.values()` | `.items()` | `.keys()` | — | N/A | N/A | `pd.Series()` | `pd.DataFrame()` |
| **array** | `list()` | `tuple()` | `set()` | `enumerate` | — | `np.array()` | `pd.Series()` | `pd.DataFrame()` |
| **numpy** | `.tolist()` | `tuple()` | `set()` | `enumerate` | `array()` | — | `pd.Series()` | `pd.DataFrame()` |
| **Series** | `.tolist()` | `tuple()` | `set()` | `.to_dict()` | `array()` | `.to_numpy()` | — | `.to_frame()` |
| **DataFrame** | `.values` | `itertuples` | N/A | `.to_dict()` | N/A | `.to_numpy()` | `df['col']` | — |

---

## Important Notes

### Performance Considerations
- Converting large datasets: numpy/pandas operations are generally faster
- Memory efficiency: numpy arrays are more memory-efficient than lists
- Use `.tolist()` for numpy/pandas → list conversion (faster than `list()`)

### Data Type Preservation
- Sets lose order and duplicates
- Dicts to lists lose keys (unless using `.items()`)
- Array types must be specified (e.g., 'i' for int, 'f' for float)
- NumPy maintains dtype information
- Pandas preserves column names and indices

### Common Pitfalls
- `set()` removes duplicates and doesn't preserve order
- Direct dict conversion from list needs `enumerate()` or `zip()`
- NumPy arrays require homogeneous types
- DataFrame column access: `df['col']` returns Series, `df[['col']]` returns DataFrame
- String to list: `list('abc')` → `['a', 'b', 'c']`, not `split()`

### Version Compatibility
- Python 3.7+: Regular dicts maintain insertion order
- Pandas 2.0+: Some methods have changed (check documentation)
- NumPy: `np.array()` vs `np.asarray()` (latter doesn't copy if already array)

---

## Additional Resources

For more complex conversions or specific use cases:
- NumPy documentation: https://numpy.org/doc/
- Pandas documentation: https://pandas.pydata.org/docs/
- Python collections: https://docs.python.org/3/library/collections.html
