import pytest
from tools.calculator import CalculatorTool


@pytest.fixture
def calc():
    return CalculatorTool()


class TestValidExpressions:
    @pytest.mark.asyncio
    async def test_basic_arithmetic(self, calc):
        assert await calc.execute("2 + 2") == "4"
        assert await calc.execute("10 - 3") == "7"
        assert await calc.execute("6 * 7") == "42"
        assert await calc.execute("15 / 3") == "5.0"

    @pytest.mark.asyncio
    async def test_power(self, calc):
        assert await calc.execute("2 ** 10") == "1024"

    @pytest.mark.asyncio
    async def test_modulo_and_floor_div(self, calc):
        assert await calc.execute("17 % 5") == "2"
        assert await calc.execute("17 // 5") == "3"

    @pytest.mark.asyncio
    async def test_unary(self, calc):
        assert await calc.execute("-5") == "-5"
        assert await calc.execute("+3") == "3"

    @pytest.mark.asyncio
    async def test_nested(self, calc):
        assert await calc.execute("(2 + 3) * 4") == "20"

    @pytest.mark.asyncio
    async def test_whitelisted_functions(self, calc):
        assert await calc.execute("abs(-5)") == "5"
        assert await calc.execute("round(3.14159, 2)") == "3.14"
        assert await calc.execute("min(1, 2, 3)") == "1"
        assert await calc.execute("max(1, 2, 3)") == "3"
        assert await calc.execute("pow(2, 10)") == "1024"


class TestDangerousInputsRejected:
    @pytest.mark.asyncio
    async def test_import(self, calc):
        result = await calc.execute("__import__('os').system('ls')")
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_class_introspection(self, calc):
        result = await calc.execute("().__class__.__bases__[0].__subclasses__()")
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_eval_call(self, calc):
        result = await calc.execute("eval('1+1')")
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_open_file(self, calc):
        result = await calc.execute("open('/etc/passwd')")
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_list_comprehension(self, calc):
        result = await calc.execute("[x for x in range(10)]")
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_attribute_access(self, calc):
        result = await calc.execute("''.__class__")
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_string_literal(self, calc):
        result = await calc.execute("'hello'")
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_variable_reference(self, calc):
        result = await calc.execute("x + 1")
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_lambda(self, calc):
        result = await calc.execute("(lambda: 1)()")
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_division_by_zero(self, calc):
        result = await calc.execute("1 / 0")
        assert result.startswith("Error:")
