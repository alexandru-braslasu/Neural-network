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
## @deftypefn {} {@var{retval} =} cost_function (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Alexandru Braslasu <Alexandru Braslasu@DESKTOP-ANP6DNN>
## Created: 2023-05-01

function [J, grad] = cost_function (params, X, y, lambda, input_layer_size, hidden_layer_size, output_layer_size)
  ##salvam teta1 si teta2 folosind functia indicata in enunt
  teta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
  teta2 = reshape(params((1 + (hidden_layer_size * (input_layer_size + 1))):end), output_layer_size, (hidden_layer_size + 1));
  m = size(X, 1);
  ##algoritmul de forward propagation
  mat = [ones(1, m); X'];
  a2 = sigmoid(teta1 * mat);
  [p, q] = size(a2);
  a2 = [ones(1, q); a2];
  a3 = sigmoid(teta2 * a2);
  ##masca cu cele 10 valori de 0 si 1
  mask = zeros(10, m);
  mask(sub2ind([10, m], y', 1 : m)) = 1;
  d3 = a3 - mask;
  ##obtinerea lui s pentru valoarea lui J
  s = -(mask .* log(a3)) - (1 - mask) .* log(1 - a3);
  delta2 = d3 * a2';
  d2 = (teta2' * d3) .* (a2 .* (1 - a2));
  d2 = d2(2 : p + 1, :);
  delta1 = d2 * mat';
  [n1, m1] = size(delta1);
  [n2, m2] = size(delta2);
  J = (lambda * (sum(sum(teta1(:, 2:m1) .^ 2)) + sum(sum(teta2(:, 2:m2) .^ 2))) + 2 * sum(s(:))) / (2 * m);
  grad = [delta1(:); delta2(:)] / m;
  grad += (lambda/m) * [zeros(n1, 1); teta1(:,2:m1)(:).'(:); zeros(n2, 1); teta2(:,2:m2)(:).'(:)];
endfunction
