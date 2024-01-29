import pandas as pd
import pickle
import os
import json
import argparse
import numpy as np


def clean_input(prompt, options, default=None):
    i = None
    options_lower = [x.lower() for x in options]
    while i is None:
        default_string = ''
        if default is not None:
            default_string = f' (DEFAULT: {default})'
        temp = input(prompt + ' [' + ','.join(options) + ']' + default_string + ': ')
        temp = temp.lower()
        if len(temp) == 0 and default is not None:
            i = default
            continue
        if temp in options_lower:
            i = options[options_lower.index(temp)]
        elif len([x for x in options_lower if x.startswith(temp)]) == 1:
            filtered = [x for x in options_lower if x.startswith(temp)]
            i = options[options_lower.index(filtered[0])]
    return i


def choose_single(options, default=None):
    options = sorted(list(map(str, options)))
    done = False
    chosen = default
    while not done:
        for i, opt in enumerate(options):
            if opt == default:
                print(f'{i+1} - {opt} <= DEFAULT')
            else:
                print(f'{i+1} - {opt}')
        temp = input(f'choose an option: ')
        if len(temp) == 0 and chosen is not None:
            done = True
            continue
        try:
            indx = int(temp) - 1
            if indx < len(options):
                chosen = options[indx]
                done = True
        except:
            pass
    return chosen


def choose_multiple(options, break_string='q', default=[]):
    options = sorted(list(map(str, options)))
    done = False
    chosen = default
    while not done:
        for i, opt in enumerate(options):
            chosen_flag = ' '
            if options[i] in chosen:
                chosen_flag = 'x'
            print(f'{i+1} [{chosen_flag}] - {opt}')
        temp = input(f'choose an option (`{break_string}` to quit, `a` for all): ')
        if temp == break_string or len(temp) == 0:
            done = True
        elif temp == 'a':
            chosen = set(options)
            done = True
        else:
            try:
                indx = int(temp) - 1
                if indx < len(options):
                    if indx in chosen:
                        chosen.remove(options[indx])
                    else:
                        chosen.append(options[indx])
            except:
                pass
    return list(chosen)


def update_mask_dropable_by_config(col, data, config, mask):
    '''
    Update row mask if the given column has categorical data marked "drop" by config
    '''
    values = data[col].values

    if col not in config['cat_cols']:
        return mask

    options = config['cat_cols'][col][0]# + config['map_to_empty'][col]

    if config['cat_cols'][col][1] != "drop":
        return mask

    new_mask = np.array([x in options for x in values])

    mask = mask * new_mask
    return mask


def update_mask_dropable_by_count(col, data, mask, min_count=3):

    values = data[col].values
    options = list(set(values))

    updated = False
    for o in options:
        if o in config['cat_cols']:
            count = (values == o)[mask].sum()
            if count < min_count:
                updated = True
                mask = mask * (values == o)

    return updated, mask


def check_for_signal(col, data):

    values = data[col].values

    if col not in config['cat_cols']:
        return True

    # Filter out empty values
    # values = [x for x in values if x and x not in config['map_to_empty'][col]]
    values = [x for x in values if x]# and not np.isnan(x)]

    # If only one class is left, we drop
    if len(set(values)) == 1:
        return False
    return True


def check_for_warn(col, data, config):

    values = data[col].values

    if col not in config['cat_cols']:
        return

    options = config['cat_cols'][col][0]# + config['map_to_empty'][col]
    # print(set(values))
    # print(options)
    options_str = ", ".join(map(lambda x: "`"+x+"`", options))

    if config['cat_cols'][col][1] != "warn":
        return

    for x in set(values):
        if str(x) not in options:
            print(f' !!! - In column `{col}` unknown value `{x}` found. Valid options are {options_str}')


def binary_encode(values):
    uniq = sorted(list(set(values)))
    result = np.array(values == uniq[0]).astype(int)
    return [result]

def trinary_encode(values, col, how='signal', s_prefix='N'):
    uniq = sorted(list(set(values)))
    if how == 'signal':
        s = [x for x in uniq if x.startswith(s_prefix)]
        if len(s) == 1:
            sig = s[0]
        else:
            sig = uniq[0]
        alt = [x for x in uniq if x != sig]
        bit0 = np.array(values != sig).astype(int)
        bit1 = np.array(values == alt[0]).astype(int) - np.array(values != sig).astype(int)
    return [f'{col}-{sig}', f'{col}-{alt[0]}_vs_{alt[1]}'], [bit0, bit1]

def one_hot_encode(values, col):
    uniq = sorted(list(set(values)))
    names = []
    bits = []
    for u in uniq:
        names.append(f'{col}-{u}')
        bits.append(np.array(values != u).astype(int))
    return names, bits

def bit_encode(data, col):
    uniq = set(data[col])
    if len(uniq) == 2:
        return [col], binary_encode(data[col])
    elif len(uniq) == 3:
        return trinary_encode(data[col], col)
    else:
        return one_hot_encode(data[col], col)

def parse_data(data, config):
    # First we scrap the columns we don't care about
    columns = [c for c in data.columns if c not in config['dropped']]
    print(f'Data has {len(data)} rows and {len(columns)} columns.')

    # make our index mask
    print('Creating mask to determine which rows can be dropped')
    mask = np.array([True] * len(data))

    # Loop over columns and drop entries that are marked "drop" in the config
    for col in columns:
        if col in config['cat_cols'] and config['cat_cols'][col][1] == "drop":
            print(f'checking {col} for entries we can drop...')
            mask = update_mask_dropable_by_config(col, data, config, mask)

    print(f'Currently {sum(mask == False)} rows are being dropped')
    # Now we drop entries that are rare
    col = 0
    updated = False
    print('=====================')
    print('Checking for entries that are droppable by being rare...')
    while True:
        # print(f'checking {columns[col]} for dropable by count... [{updated}]')
        # Our update mask function will return `True` if the mask was updated
        check, mask = update_mask_dropable_by_count(columns[col], data, mask)
        if check:
            print(f' !!! - found dropable entries in {columns[col]}')
        updated = updated or check
        col += 1
        # When we get to the last column
        if col == len(columns):
            # if the mask wasn't updated...
            if not updated:
                # We're done
                break
            # Otherwise...
            else:
                # Reset the `updated` flag
                updated = False
                # and go back to the start
                col = 0

    # Now we filter the data with the mask we've created
    print(f'{sum(mask == False)} rows are being dropped')
    data = data[mask]

    print('=======================')
    print('Checking for unknown values that require warnings...')
    for col, details in config['cat_cols'].items():
        if details[1] == "warn":
            # print(f'Checking {col} for unknown values')
            check_for_warn(col, data, config)

    print('=======================')
    print('Checking for columns that lack signal...')
    cols_to_drop = []
    for col in columns:
        if not check_for_signal(col, data):
            cols_to_drop.append(col)
            print(f"dropping column `{col}` due to lack of signal")

    columns = [x for x in columns if x not in cols_to_drop]

    print('=======================')
    print(f'Data has {len(data)} rows and {len(columns)} columns.')
    data = data[columns]

    encoded = []
    labels = []
    for col in data.columns:
        if col in config['cat_cols']:
            l, d = bit_encode(data, col)
            encoded += d
            labels += l
    # print(labels)
    # print(encoded)
    # print(np.array(encoded))
    encoded_data = pd.DataFrame(np.array(encoded).T, columns=labels)
    with open('encoded_data.csv', 'w') as f:
        encoded_data.to_csv(f, index=False)

def generate_config(data, skip=False, config_file='config.json'):
    dropped_cols = set()
    col_types = {}
    cat_cols = {}
    # map_to_empty = {}
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            prior = json.load(f)
            dropped_cols = set(prior['dropped'])
            col_types = prior['col_types']
            cat_cols = prior['cat_cols']
            # map_to_empty = {k: set(v) for k, v in prior['map_to_empty'].items()}

    for col in data.columns:
        if skip and ((col in dropped_cols) or (col in col_types)):
            continue
        print(col)
        print()
        col_data = data[col]
        for x in list(set(map(str, col_data.values)))[:10]:
            print(x, list(map(str, col_data.values)).count(x))
        print()

        default = 'y'
        if col in col_types:
            default = 'y'
        elif col in dropped_cols:
            default = 'n'
        if clean_input('keep? ', ['y', 'n'], default=default) == 'y':
            # We keep it

            default = None
            if col in col_types:
                default = col_types[col]

            col_type = choose_single(['Categorical', 'Binned'], default=default)
            col_types[col] = col_type
            if col_type == 'Categorical':
                options = list(set(map(str, col_data.values)))
                default = [[], None]
                if col in cat_cols:
                    default = cat_cols[col]
                    if len(default) != 2:
                        default = [[], None]

                # print('Choose categories to consider')
                # categories = choose_multiple(options, default=default[0])
                categories = options
                # print('What to do with unmatched values?')
                # unmatched = choose_single(['keep', 'drop', 'warn', 'random', 'major'], default=default[1])
                unmatched = 'warn'
                cat_cols[col] = (categories, unmatched)

                # default = []
                # if col in map_to_empty:
                #     default = map_to_empty[col]
                # print('Which categories are "empty"?')
                # for c in categories:
                #     options.remove(c)
                # categories = choose_multiple(options, default=default)
                # map_to_empty[col] = categories
            if col_type == 'Binned':
                pass
                # # Should the user get to provide bins here?
                # default = []
                # # if col in map_to_empty:
                # #     default = map_to_empty[col]
                # print('Which categories are "empty"?')
                # options = set(map(str, col_data.values))
                # categories = choose_multiple(options, default=default)
                # # map_to_empty[col] = categories
        else:
            dropped_cols.add(col)

        with open(config_file, 'w') as f:
            json.dump({
                        'dropped': list(dropped_cols),
                        'col_types': col_types,
                        'cat_cols': cat_cols,
                        # 'map_to_empty': {k: list(v) for k, v in map_to_empty.items()}},
                      },
                      f, indent=2)
        # print('dropped:', dropped_cols)
        # print('col_types:', col_types)
        # print('cat_cols:', cat_cols)
        print('========================')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='File to load')
    parser.add_argument('--config_file', help='Config file to use', nargs=1, default='config.json')
    parser.add_argument('--sheet', help='which sheet(s) to read if loading an excel file')
    parser.add_argument('--skip', help='Skip entries that we already configured', action='store_true')
    parser.add_argument('--config', help='Generate config', action='store_true')
    parser.add_argument('--parse', help='Parse from config', action='store_true')

    args = parser.parse_args()

    print(args.file)

    if args.file.endswith('csv'):
        with open(args.file, 'r') as f:
            data = pd.read_csv(f)
    elif args.file.endswith('xlsx'):
        with open(args.file, 'rb') as f:
            data = pd.read_excel(f, sheet_name=args.sheet)
            data = data.fillna('None')
    else:
        print('We don\'t know how to load the provided file')
        exit

    if args.config:
        generate_config(data, args.skip, args.config_file)

    if args.parse:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        parse_data(data, config)
