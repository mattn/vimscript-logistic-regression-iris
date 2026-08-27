function! s:softmax(w, x) abort
  let l:v = a:w[0] * a:x[0] + a:w[1] * a:x[1]
        \ + a:w[2] * a:x[2] + a:w[3] * a:x[3]
  return 1.0 / (1.0 + exp(-l:v))
endfunction

function! s:predict(w, x) abort
    return s:softmax(a:w, a:x)
endfunction

function! s:logistic_regression(X, y, rate, ntrains) abort
  let l:w0 = (rand() / 4294967295.0 - 0.5) * 2
  let l:w1 = (rand() / 4294967295.0 - 0.5) * 2
  let l:w2 = (rand() / 4294967295.0 - 0.5) * 2
  let l:w3 = (rand() / 4294967295.0 - 0.5) * 2
  let l:rate = a:rate
  let l:samples = map(copy(a:X), 'v:val + [a:y[v:key]]')
  for l:n in range(a:ntrains)
    for l:x in l:samples
      let [l:x0, l:x1, l:x2, l:x3, l:y] = l:x
      let l:v = l:w0 * l:x0 + l:w1 * l:x1
            \ + l:w2 * l:x2 + l:w3 * l:x3
      let l:pred = 1.0 / (1.0 + exp(-l:v))
      let l:scale = l:rate * (l:y - l:pred) * l:pred
            \ * (1.0 - l:pred) * 4
      let l:w0 += l:x0 * l:scale
      let l:w1 += l:x1 * l:scale
      let l:w2 += l:x2 * l:scale
      let l:w3 += l:x3 * l:scale
    endfor
  endfor
  return [l:w0, l:w1, l:w2, l:w3]
endfunction

function! s:token(line) abort
  return map(split(a:line, ','), 'v:val =~# "^[-+]\\?[0-9][.]\\?[0-9]*$" ? str2float(v:val) : v:val')
endfunction

function! s:make_vocab(names) abort
  let l:ns = {}
  for l:name in a:names
    if !has_key(l:ns, l:name)
      let l:ns[l:name] = 0.0 + len(l:ns)
    endif
  endfor
  return l:ns
endfunction

function! s:bag_of_words(names, vocab) abort
  return map(a:names, '(0.0 + a:vocab[v:val]) / (len(a:vocab) - 1)')
endfunction

function! s:shuffle(arr)
  let l:arr = a:arr
  let l:i = len(l:arr)
  while l:i
    let l:i -= 1
    let l:j = float2nr(rand() / 4294967295.0 * l:i) % len(l:arr)
    if l:i ==# l:j
      continue
    endif
    let [l:arr[l:i], l:arr[l:j]] = [l:arr[l:j], l:arr[l:i]]
  endwhile
  return l:arr
endfunction

function! s:main() abort
  let l:data = map(readfile('iris.csv'), 's:token(v:val)')[1:]
  call s:shuffle(l:data)
  let [l:train, l:test] = [l:data[:len(data)/2], l:data[len(data)/2+1:]]

  let [l:X, l:y] = [[], []]
  for l:row in l:train
    call add(l:X, l:row[:3])
    call add(l:y, l:row[4])
  endfor
  let l:vocab = s:make_vocab(l:y)
  call s:bag_of_words(l:y, l:vocab)
  let l:ni = map(sort(map(keys(l:vocab), '[v:val, float2nr(l:vocab[v:val])]'), {a, b -> a[1] - b[1]}), 'v:val[0]')

  let l:w = s:logistic_regression(l:X, l:y, 0.01, 5000)

  let l:count = 0
  let l:size = len(l:vocab) - 1
  for l:row in l:test
    let l:r = s:predict(l:row[:3], l:w)
    if l:ni[min([float2nr(l:r * l:size + 0.1), l:size])] ==# l:row[4]
      let l:count += 1
    endif
  endfor
  echo (0.0 + l:count) / len(l:test)
endfunction

function! s:benchmark()
  let l:start = reltime()
  call s:main()
  echomsg str2float(reltimestr(reltime(l:start)))
endfunction

call s:benchmark()
