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
## @deftypefn {} {@var{retval} =} split_dataset (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Alexandru Braslasu <Alexandru Braslasu@DESKTOP-ANP6DNN>
## Created: 2023-05-01

function [X_train, y_train, X_test, y_test] = split_dataset(X, y, percent)
  [m, n] = size(X);
  indici = randperm(size(X));
  X_amestecat = X(indici, :);
  y_amestecat = y(indici);
  m_nou = percent * m;;
  X_train = X_amestecat(1 : m_nou, 1 : n);
  X_test = X_amestecat((m_nou + 1) : m, 1 : n);
  y_train = y_amestecat(1 : m_nou);
  y_test = y_amestecat((m_nou+1) : m);
endfunction
