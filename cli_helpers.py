
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