from typing import Tuple, Union, List
from contextlib import contextmanager
# def type alias
MonomialType = Union[int, Tuple[int], 'Monomial']
PolynomialType = Union[MonomialType, 'Polynomial']

__all__ = ['Monomial', 'Polynomial', 'MonomialType', 'PolynomialType', 'modulo_context', 'set_modulo', 'get_modulo']

from contextlib import contextmanager
from typing import Tuple

# Global properties to set the modulo for coefficients
CMODULO: int = 0
XMODULO: int = 0
YMODULO: int = 0

@contextmanager
def modulo_context(c_modulo: int, x_modulo: int, y_modulo: int):
    """
    A context manager to manage the modulo for coefficients.

    Args:
        c_modulo (int): The new value for CMODULO.
        x_modulo (int): The new value for XMODULO.
        y_modulo (int): The new value for YMODULO.
    """
    global CMODULO, XMODULO, YMODULO
    original_cmodulo, original_xmodulo, original_ymodulo = CMODULO, XMODULO, YMODULO

    try:
        CMODULO = c_modulo
        XMODULO = x_modulo
        YMODULO = y_modulo
        yield
    finally:
        CMODULO, XMODULO, YMODULO = original_cmodulo, original_xmodulo, original_ymodulo

def set_modulo(c_modulo: int, x_modulo: int, y_modulo: int) -> None:
    """Set the modulo for coefficients."""
    global CMODULO, XMODULO, YMODULO
    CMODULO = c_modulo
    XMODULO = x_modulo
    YMODULO = y_modulo

def get_modulo() -> Tuple[int, int, int]:
    """Get the modulo for coefficients."""
    return CMODULO, XMODULO, YMODULO

def _cast_to_monomial(other: MonomialType) -> 'Monomial':
    if isinstance(other, int):
        return Monomial((other, 0, 0))
    elif isinstance(other, tuple):
        return Monomial(other)
    elif isinstance(other, Monomial):
        return other
    else:
        raise ValueError("Invalid input for Monomial")

def _cast_from_polynomial(other: 'Polynomial') -> PolynomialType:
    if isinstance(other, Polynomial):
        if len(other.terms) == 0:
            return Monomial(0)
        elif len(other.terms) == 1:
            return other.terms[0]
        else:
            return other
    else:
        raise ValueError("Invalid input for Polynomial")

class Monomial:
    def __new__(cls, expr: Union[Tuple[int],str, int] = None) -> 'Monomial':
        if expr is None:
            return cls((0, 0, 0))
        elif isinstance(expr, int):
            return cls((expr, 0, 0))
        elif isinstance(expr, tuple):
            return cls._create_from_tuple(expr)
        elif isinstance(expr, str):
            return cls._create_from_str(expr)
        else:
            raise ValueError("Invalid input for Monomial")

    @classmethod
    def _create_from_tuple(cls, expr: Tuple[int]) -> 'Monomial':
        coeff, x_exp, y_exp = expr
        instance = super().__new__(cls)
        instance.coefficient = coeff
        instance.x_exponent = x_exp
        instance.y_exponent = y_exp
        instance.simplify()
        return instance

    @classmethod
    def _create_from_str(cls, expr: str) -> 'Monomial':
        # Implement the parsing logic here
        pass

    def simplify(self) -> None:
        """Simplify the monomial by modulo the coefficients and the exponents."""
        CMODULO, XMODULO, YMODULO = get_modulo()
        self.coefficient = self.coefficient % CMODULO if CMODULO else self.coefficient
        self.x_exponent = self.x_exponent % XMODULO if XMODULO else self.x_exponent
        self.y_exponent = self.y_exponent % YMODULO if YMODULO else self.y_exponent

    def __add__(self, other: PolynomialType) -> PolynomialType:
        if isinstance(other, Polynomial):
            # polynomial addition
            return other + self
        else:
            other = _cast_to_monomial(other)
        # monomial addition
        if self.x_exponent == other.x_exponent and self.y_exponent == other.y_exponent:
            return Monomial((self.coefficient + other.coefficient, self.x_exponent, self.y_exponent))
        else:
            return Polynomial([self, other])

    def __sub__(self, other: PolynomialType) -> PolynomialType:
        if isinstance(other, Polynomial):
            # polynomial subtraction
            return -other + self
        else:
            other = _cast_to_monomial(other)
        # monomial subtraction
        if self.x_exponent == other.x_exponent and self.y_exponent == other.y_exponent:
            return Monomial((self.coefficient - other.coefficient, self.x_exponent, self.y_exponent))
        else:
            return Polynomial([self, -other])

    def __mul__(self, other: PolynomialType) -> PolynomialType:
        if isinstance(other, Polynomial):
            # polynomial multiplication
            return other * self
        else:
            other = _cast_to_monomial(other)
        # monomial multiplication
        return Monomial((self.coefficient * other.coefficient, self.x_exponent + other.x_exponent, self.y_exponent + other.y_exponent))

    def __truediv__(self, other: MonomialType) -> 'Monomial':
        other = _cast_to_monomial(other)
        if other.coefficient == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return Monomial((self.coefficient / other.coefficient, self.x_exponent - other.x_exponent, self.y_exponent - other.y_exponent))

    def __neg__(self) -> 'Monomial':
        return Monomial((-self.coefficient, self.x_exponent, self.y_exponent))

    def __abs__(self) -> 'Monomial':
        return Monomial((abs(self.coefficient), self.x_exponent, self.y_exponent))

    def __eq__(self, other: MonomialType) -> bool:
        if isinstance(other, int):
            return (self.coefficient, self.x_exponent, self.y_exponent) == (other, 0, 0)
        else:
            return (self.coefficient, self.x_exponent, self.y_exponent) == (other.coefficient, other.x_exponent, other.y_exponent)

    def __lt__(self, other: MonomialType) -> bool:
        other = _cast_to_monomial(other)

        if self.x_exponent != other.x_exponent:
            return self.x_exponent < other.x_exponent
        elif self.y_exponent != other.y_exponent:
            return self.y_exponent < other.y_exponent
        else:
            return self.coefficient < other.coefficient

    def __int__(self) -> int:
        if self.x_exponent == 0 and self.y_exponent == 0:
            return self.coefficient
        else:
            raise ValueError("Only scalar term can be converted to integers")

    def __iter__(self):
        return iter([self])

    def conj(self) -> 'Monomial':
        return Monomial((self.coefficient, -self.x_exponent, -self.y_exponent))
    
    @property
    def T(self) -> 'Monomial':
        return self.conj()

    def __str__(self) -> str:
        terms = []
        if self.coefficient != 0:
            if self.coefficient != 1 or (self.x_exponent == 0 and self.y_exponent == 0):
                terms.append(str(self.coefficient))
        else:
            return "0"
        if self.x_exponent != 0:
            terms.append(f"x^{self.x_exponent}")
        if self.y_exponent != 0:
            terms.append(f"y^{self.y_exponent}")
        return "".join(terms)

    def __repr__(self) -> str:
        return f"Monomial({self.coefficient}, {self.x_exponent}, {self.y_exponent})"

    def __hash__(self) -> int:
        return hash((self.coefficient, self.x_exponent, self.y_exponent))

class Polynomial:
    def __new__(cls, expr: Union[List[Monomial], List[Tuple[int]], str] = None) -> Union['Polynomial', 'Monomial', int]:
        if expr is None:
            return cls([])
        elif isinstance(expr, list):
            expr = [_cast_to_monomial(term) for term in expr]
            if all(isinstance(term, Monomial) for term in expr):
                instance = cls._create_from_monomials(expr)
            else:
                raise ValueError("Invalid input for Polynomial")
        elif isinstance(expr, str):
            instance = cls._create_from_str(expr)
        else:
            raise ValueError("Invalid input for Polynomial")
        
        return _cast_from_polynomial(instance) # downgrad to Monomial if possible

    @classmethod
    def _create_from_monomials(cls, expr: List[Monomial]) -> PolynomialType:
        instance = super().__new__(cls)
        instance.terms = expr
        instance.simplify()
        return instance

    @classmethod
    def _create_from_str(cls, expr: str) -> 'Polynomial':
        # Implement the parsing logic here
        pass


    def simplify(self) -> None:
        """Simplify the polynomial by combining like terms and sorting."""
        self.terms.sort()
        i = 0
        while i < len(self.terms) - 1:
            if self.terms[i].x_exponent == self.terms[i+1].x_exponent and self.terms[i].y_exponent == self.terms[i+1].y_exponent:
                self.terms[i] = Monomial((self.terms[i].coefficient + self.terms[i+1].coefficient, self.terms[i].x_exponent, self.terms[i].y_exponent))
                self.terms.pop(i+1)
            else:
                i += 1
        self.terms = [term for term in self.terms if term.coefficient != 0]

    def __add__(self, other: PolynomialType) -> PolynomialType:
        if isinstance(other, Polynomial):
            return Polynomial(self.terms + other.terms)
        else:
            other = _cast_to_monomial(other)
            return Polynomial(self.terms + [other])

    def __sub__(self, other: PolynomialType) -> PolynomialType:
        return self + (-other)

    def __mul__(self, other: PolynomialType) -> PolynomialType:
        if isinstance(other, Polynomial):
            result = []
            for a in self.terms:
                for b in other.terms:
                    result.append(a * b)
            return Polynomial(result)
        else:
            other = _cast_to_monomial(other)
            return Polynomial([term * other for term in self.terms])

    def __truediv__(self, other: PolynomialType) -> PolynomialType:
        if isinstance(other, Polynomial):
            # Implement polynomial division
            pass
            raise NotImplementedError("Polynomial division is not implemented")
        else:
            other = _cast_to_monomial(other)
            return Polynomial([term / other for term in self.terms])

    def __neg__(self) -> 'Polynomial':
        return Polynomial([-term for term in self.terms])
    
    def __abs__(self) -> 'Polynomial':
        return Polynomial([abs(term) for term in self.terms])
    
    def __eq__(self, other: PolynomialType) -> bool:
        if isinstance(other, Polynomial):
            return self.terms == other.terms
        else:
            return False
    
    def __iter__(self):
        return iter(self.terms)

    def conj(self) -> 'Polynomial':
        return Polynomial([term.conj() for term in self.terms])
    
    @property
    def T(self) -> 'Polynomial':
        return self.conj()

    def __str__(self) -> str:
        terms = []
        for term in self.terms:
            term_str = str(term)
            if term.coefficient < 0:
                term_str = f"- {abs(term)}"
            else:
                if len(terms) > 0:
                    term_str = f"+ {term_str}"
            terms.append(term_str)
        return " ".join(terms)

    def __repr__(self) -> str:
        return f"Polynomial({self.terms})"
    
    def __hash__(self) -> int:
        return hash(tuple(self.terms))