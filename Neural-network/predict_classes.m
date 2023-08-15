## Copyright (C) 2023 Alexandru Braslasu
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {} {@var{retval} =} predict_classes (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Alexandru Braslasu <Alexandru Braslasu@DESKTOP-ANP6DNN>
## Created: 2023-05-07

function [classes] = predict_classes (X, weights, input_layer_size, hidden_layer_size, output_layer_size)
  teta1 = reshape(weights(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
  teta2 = reshape(weights((1 + (hidden_layer_size * (input_layer_size + 1))):end), output_layer_size, (hidden_layer_size + 1));
  m = size(X, 1);
  mat = [ones(1, m); X'];
  a2 = sigmoid(teta1 * mat);
  [p, q] = size(a2);
  a2 = [ones(1, q); a2];
  a3 = sigmoid(teta2 * a2);
  classes = zeros(m, 1);
  for i = 1 : m
    maxi = -1;
    for j = 1 : 10
      if (a3(j, i) > maxi)
        maxi = a3(j, i);
        val = j;
      endif
    endfor
    classes(i) = val;
  endfor
endfunction
