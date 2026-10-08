from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

np.set_printoptions(precision=5, suppress=True)

ArrayLike = Sequence[float]


def format_number(value: float, precision: int = 5) -> str:
    formatted = f"{value:.{precision}f}"
    if formatted.endswith("." + "0" * precision):
        formatted = formatted[: -(precision + 1)]
    if formatted == "-0":
        formatted = "0"
    return formatted


def format_signed_number(value: float, precision: int = 5) -> str:
    core = format_number(value, precision)
    if core == "0":
        return core
    if core.startswith("-"):
        return core
    return "+" + core



# Reporter: controla saída em texto ou em LaTeX
class Reporter:
    def __init__(
        self,
        mode: str = "text",
        report_file: Optional[str] = None,
        continue_document: bool = False,
        finalize_document: bool = True,
    ) -> None:
        self.mode = mode.lower()
        self.lines: List[str] = []
        self.report_file = report_file
        self.continue_document = continue_document and self.mode == "latex"
        self._started_doc = self.continue_document
        self.finalize_document = finalize_document

    def _ensure_doc_started(self) -> None:
        if self.mode == "latex" and self.report_file and not self._started_doc:
            self.lines.append(r"\documentclass[11pt,a4paper]{article}")
            self.lines.append(r"\usepackage{amsmath, amssymb, bm}")
            self.lines.append(r"\usepackage[margin=2.5cm]{geometry}")
            self.lines.append(r"\begin{document}")
            self._started_doc = True

    def title(self, text: str) -> None:  # type: ignore[name-defined]
        if self.mode == "latex":
            self._ensure_doc_started()
            self.lines.append(f"\\section*{{{text}}}")
            self.lines.append(r"\medskip")
        else:
            self.lines.append(f"\n=== {text} ===")

    def text(self, s: str, latex_break: bool = True) -> None:
        if self.mode == "latex":
            self._ensure_doc_started()
            stripped = s.strip()
            if not stripped:
                self.lines.append("")
                return
            content = stripped
            if latex_break:
                self.lines.append(r"\noindent " + content + r"\\")
                self.lines.append(r"\smallskip")
            else:
                self.lines.append(content)
        else:
            self.lines.append(s)

    def eq(self, content: str, numbered: bool = False) -> None:
        if self.mode == "latex":
            self._ensure_doc_started()
            if numbered:
                self.lines.append("\\begin{equation}\n" + content + "\n\\end{equation}")
            else:
                self.lines.append("\\[" + content + "\\]")
            self.lines.append(r"\smallskip")
        else:
            self.lines.append(content)

    def block(self, content: str) -> None:
        if self.mode == "latex":
            self._ensure_doc_started()
            self.lines.append("\\begin{align*}\n" + content + "\n\\end{align*}")
            self.lines.append(r"\smallskip")
        else:
            cleaned = content.replace("\\\\\n", "\n").replace("\\\\", "\n")
            self.lines.append(cleaned)

    def flush(self) -> None:
        if self.mode == "latex" and self.report_file:
            if self._started_doc and self.finalize_document:
                self.lines.append(r"\end{document}")
            write_mode = "a" if self.continue_document else "w"
            with open(self.report_file, write_mode, encoding="utf-8") as f:
                content = "\n".join(self.lines)
                if self.continue_document and content:
                    f.write("\n")
                f.write(content)

    def dumps(self) -> str:
        return "\n".join(self.lines)

# TransferFunction e utilitários
def as_array(data: ArrayLike) -> np.ndarray:
    return np.array(list(data), dtype=float)


def format_complex_str(z: complex) -> str:
    real = z.real
    imag = z.imag
    if math.isclose(imag, 0.0, abs_tol=1e-12):
        return format_number(real)
    sign = "+" if imag >= 0 else "-"
    return f"{format_number(real)} {sign} j{format_number(abs(imag))}"


def format_complex_latex(z: complex) -> str:
    return format_complex_str(z).replace(" j", r"\,j")


def latex_align(*lines: str) -> str:
    stripped = [line.rstrip() for line in lines]
    if not stripped:
        return ""
    formatted: List[str] = []
    last_idx = len(stripped) - 1
    for idx, line in enumerate(stripped):
        if idx < last_idx:
            formatted.append(f"{line} \\\\ ")
        else:
            formatted.append(line)
    return "\n".join(formatted)


def poly_to_latex(coeffs: np.ndarray, var: str = "s") -> str:
    terms: List[str] = []
    degree = len(coeffs) - 1
    for idx, coef in enumerate(coeffs):
        power = degree - idx
        if abs(coef) < 1e-12:
            continue
        coef_str = format_number(coef)
        if power == 0:
            terms.append(coef_str)
        elif power == 1:
            terms.append(f"{coef_str}{var}")
        else:
            terms.append(f"{coef_str}{var}^{power}")
    if not terms:
        return "0"
    poly_expr = " + ".join(terms)
    return poly_expr.replace("+ -", "- ")


def format_complex_list_latex(values: Sequence[complex]) -> str:
    if not values:
        return r"\varnothing"
    return r"\{ " + ", ".join(format_complex_latex(complex(v)) for v in values) + r" \}"


def format_complex_list_text(values: Sequence[complex]) -> str:
    if not values:
        return "{}"
    return "{ " + ", ".join(format_complex_str(complex(v)) for v in values) + " }"


def format_gain_value(value: Optional[float]) -> str:
    if value is None:
        return r"\text{N/A}"
    return format_number(value)


def format_array_latex(arr: np.ndarray) -> str:
    return r"\left[" + ", ".join(format_number(float(coef)) for coef in arr) + r"\right]"


def describe_controller_structure(
    rep: Reporter,
    controller_type: str,
    pid_equal_zeros: bool,
    pid_zero_split_deg: Optional[Tuple[float, float]],
) -> None:
    controller_type = controller_type.lower()
    if rep.mode == "latex":
        rep.text(r"\textbf{Estrutura nominal do controlador}")
        if controller_type == "pd":
            rep.block(
                latex_align(
                    r"G_c(s) = K_p + K_d s",
                    r"= K_d \bigl(s + \frac{K_p}{K_d}\bigr)",
                    r"\Rightarrow z_c = -\frac{K_p}{K_d}"
                )
            )
            rep.text(r"O zero $z_c$ sera ajustado na etapa de angulo.")
        elif controller_type == "pi":
            rep.block(
                latex_align(
                    r"G_c(s) = K_p + \dfrac{K_i}{s}",
                    r"= \dfrac{K_p s + K_i}{s} = K_p \dfrac{s + \frac{K_i}{K_p}}{s}",
                    r"\Rightarrow z_c = -\frac{K_i}{K_p},\; p_c = 0"
                )
            )
            rep.text(r"O zero $z_c$ sera determinado pela condicao de angulo.")
        elif controller_type == "pid":
            rep.block(
                latex_align(
                    r"G_c(s) = K_p + \dfrac{K_i}{s} + K_d s",
                    r"= \dfrac{K_d s^2 + K_p s + K_i}{s}",
                    r"= K_d \dfrac{(s + z_1)(s + z_2)}{s}",
                    r"z_1 + z_2 = \dfrac{K_p}{K_d},\quad z_1 z_2 = \dfrac{K_i}{K_d}"
                )
            )
            if pid_equal_zeros:
                rep.text(r"Nesta execucao adotamos $z_1 = z_2 = z_c$.")
            elif pid_zero_split_deg is not None:
                phi1, phi2 = pid_zero_split_deg
                rep.text(
                    fr"Dividimos o deficit de angulo: \(\varphi_1 = {format_number(phi1)}^\circ\), \(\varphi_2 = {format_number(phi2)}^\circ\)."
                )
            else:
                rep.text(r"Os zeros $z_1$ e $z_2$ serao obtidos na etapa de angulo.")
        else:
            rep.text(r"Tipo de controlador nao listado para detalhamento.")
    else:
        rep.text("Estrutura nominal do controlador:")
        if controller_type == "pd":
            rep.text("  Gc(s) = Kp + Kd s")
            rep.text("         = Kd * (s + Kp/Kd)")
            rep.text("  => zero em s = -Kp/Kd (ajustado pela condicao de angulo)")
        elif controller_type == "pi":
            rep.text("  Gc(s) = Kp + Ki/s")
            rep.text("         = (Kp*s + Ki)/s = Kp * (s + Ki/Kp) / s")
            rep.text("  => zero em s = -Ki/Kp, polo fixo em s = 0")
        elif controller_type == "pid":
            rep.text("  Gc(s) = Kp + Ki/s + Kd s")
            rep.text("         = (Kd*s^2 + Kp*s + Ki)/s")
            rep.text("         = Kd * (s + z1)(s + z2) / s")
            rep.text("    com z1 + z2 = Kp/Kd e z1*z2 = Ki/Kd")
            if pid_equal_zeros:
                rep.text("    adotamos z1 = z2 (zero duplo determinado pela condicao de angulo)")
            elif pid_zero_split_deg is not None:
                phi1, phi2 = pid_zero_split_deg
                rep.text(
                    f"    deficit de angulo dividido em phi1={format_number(phi1)} deg e phi2={format_number(phi2)} deg"
                )
        else:
            rep.text("  Tipo de controlador nao listado para detalhamento")


def detail_open_loop_development(
    rep: Reporter,
    controller_type: str,
    plant_total: TransferFunction,
    plant_zeros: Sequence[complex],
    plant_poles: Sequence[complex],
) -> None:
    controller_type = controller_type.lower()
    if rep.mode == "latex":
        rep.text(r"\textbf{Malha aberta com o controlador escolhido}")
        rep.block(
            latex_align(
                r"L_0(s) = G(s)H(s) = \dfrac{N_{L_0}(s)}{D_{L_0}(s)}",
                fr"N_{{L_0}}(s) = {poly_to_latex(plant_total.num)}",
                fr"D_{{L_0}}(s) = {poly_to_latex(plant_total.den)}"
            )
        )
        rep.block(
            latex_align(
                fr"Z(L_0) = {format_complex_list_latex(plant_zeros)}",
                fr"P(L_0) = {format_complex_list_latex(plant_poles)}"
            )
        )
        if controller_type == "pd":
            rep.block(
                latex_align(
                    r"G_c(s) = K_c (s - z_c)",
                    r"L(s) = G_c(s) L_0(s) = K_c (s - z_c) \dfrac{N_{L_0}(s)}{D_{L_0}(s)}",
                    r"Z(L) = Z(L_0) \cup \{z_c\},\quad P(L) = P(L_0)",
                    r"1 + L(s) = 0 \Rightarrow D_{L_0}(s) + K_c (s - z_c) N_{L_0}(s) = 0"
                )
            )
        elif controller_type == "pi":
            rep.block(
                latex_align(
                    r"G_c(s) = K_c \dfrac{s - z_c}{s}",
                    r"L(s) = K_c \dfrac{s - z_c}{s} \dfrac{N_{L_0}(s)}{D_{L_0}(s)}",
                    r"Z(L) = Z(L_0) \cup \{z_c\},\quad P(L) = P(L_0) \cup \{0\}",
                    r"1 + L(s) = 0 \Rightarrow s D_{L_0}(s) + K_c (s - z_c) N_{L_0}(s) = 0"
                )
            )
        elif controller_type == "pid":
            rep.block(
                latex_align(
                    r"G_c(s) = K_c \dfrac{(s - z_1)(s - z_2)}{s}",
                    r"L(s) = K_c \dfrac{(s - z_1)(s - z_2)}{s} \dfrac{N_{L_0}(s)}{D_{L_0}(s)}",
                    r"Z(L) = Z(L_0) \cup \{z_1, z_2\},\quad P(L) = P(L_0) \cup \{0\}",
                    r"1 + L(s) = 0 \Rightarrow s D_{L_0}(s) + K_c (s - z_1)(s - z_2) N_{L_0}(s) = 0"
                )
            )
        else:
            rep.text("Detalhamento adicional nao disponibilizado para este tipo de controlador.")
    else:
        rep.text("Desenvolvimento da malha aberta com o controlador escolhido:")
        num_txt = plant_total._poly_to_string(plant_total.num, "s")
        den_txt = plant_total._poly_to_string(plant_total.den, "s")
        rep.text(f"  L0(s) = G(s)*H(s) = ({num_txt}) / ({den_txt})")
        rep.text(f"  Z(L0) = {format_complex_list_text(plant_zeros)}")
        rep.text(f"  P(L0) = {format_complex_list_text(plant_poles)}")
        if controller_type == "pd":
            rep.text("  Gc(s) = Kc*(s - zc)")
            rep.text("  L(s) = Kc*(s - zc)*L0(s)")
            rep.text("  Z(L) = Z(L0) U {zc}")
            rep.text("  P(L) = P(L0)")
            rep.text("  1 + L(s) = 0 -> D_L0(s) + Kc*(s - zc)*N_L0(s) = 0")
        elif controller_type == "pi":
            rep.text("  Gc(s) = Kc*(s - zc)/s")
            rep.text("  L(s) = Kc*(s - zc)/s * L0(s)")
            rep.text("  Z(L) = Z(L0) U {zc}")
            rep.text("  P(L) = P(L0) U {0}")
            rep.text("  1 + L(s) = 0 -> s*D_L0(s) + Kc*(s - zc)*N_L0(s) = 0")
        elif controller_type == "pid":
            rep.text("  Gc(s) = Kc*(s - z1)*(s - z2)/s")
            rep.text("  L(s) = Kc*(s - z1)*(s - z2)/s * L0(s)")
            rep.text("  Z(L) = Z(L0) U {z1, z2}")
            rep.text("  P(L) = P(L0) U {0}")
            rep.text("  1 + L(s) = 0 -> s*D_L0(s) + Kc*(s - z1)*(s - z2)*N_L0(s) = 0")
        else:
            rep.text("  Tipo de controlador nao listado para desenvolvimento detalhado")


@dataclass
class TransferFunction:
    """Simple continuous-time transfer function helper."""

    num: np.ndarray
    den: np.ndarray
    gain: float = field(init=False)

    def __post_init__(self) -> None:
        if self.den[0] == 0:
            raise ValueError("Leading denominator coefficient cannot be zero.")
        self.gain = float(self.num[0] / self.den[0])

    @classmethod
    def from_coeffs(cls, num: ArrayLike, den: ArrayLike) -> "TransferFunction":
        return cls(as_array(num), as_array(den))

    @property
    def zeros(self) -> np.ndarray:
        return np.roots(self.num) if len(self.num) > 1 else np.array([])

    @property
    def poles(self) -> np.ndarray:
        return np.roots(self.den)

    def series(self, other: "TransferFunction") -> "TransferFunction":
        num = np.polymul(self.num, other.num)
        den = np.polymul(self.den, other.den)
        return TransferFunction(num, den)

    def feedback(self, other: Optional["TransferFunction"] = None) -> "TransferFunction":
        if other is None:
            other = TransferFunction.from_coeffs([1.0], [1.0])
        loop = self.series(other)
        den_cl = poly_add(loop.den, loop.num)
        return TransferFunction(loop.num, den_cl)

    def _poly_to_string(self, coeffs: np.ndarray, var: str) -> str:
        terms: List[str] = []
        degree = len(coeffs) - 1
        for idx, coef in enumerate(coeffs):
            power = degree - idx
            if abs(coef) < 1e-12:
                continue
            coef_str = format_number(float(coef))
            if power == 0:
                terms.append(f"{coef_str}")
            elif power == 1:
                terms.append(f"{coef_str}{var}")
            else:
                terms.append(f"{coef_str}{var}^{power}")
        return " + ".join(terms) if terms else "0"

    def pprint(self, rep: Reporter, name: str = "G") -> None:
        num_str = self._poly_to_string(self.num, "s")
        den_str = self._poly_to_string(self.den, "s")
        if rep.mode == "latex":
            rep.eq(fr"{name}(s) = \dfrac{{({num_str})}}{{({den_str})}}")
        else:
            rep.text(f"{name}(s) = ({num_str}) / ({den_str})")


# ===== Helper functions =====
def poly_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    la, lb = len(a), len(b)
    if la < lb:
        a = np.r_[np.zeros(lb - la), a]
    elif lb < la:
        b = np.r_[np.zeros(la - lb), b]
    return a + b


def degrees(angle_rad: float) -> float:
    return math.degrees(angle_rad)


def angle_deg(z: complex) -> float:
    return degrees(cmath.phase(z))


def damping_from_percent_overshoot(mp_percent: float) -> float:
    if mp_percent <= 0:
        return 1.0
    mp = mp_percent / 100.0
    ln_mp = math.log(mp)
    return -ln_mp / math.sqrt(math.pi**2 + ln_mp**2)


def wn_from_settling_time(ts: float, zeta: float, criterion: str) -> float:
    if zeta <= 0:
        raise ValueError("zeta deve ser positivo para calcular t_s.")
    criterion = criterion.strip()
    if criterion == "2":
        return 4.0 / (zeta * ts)
    if criterion == "5":
        return 3.0 / (zeta * ts)
    raise ValueError("criterion deve ser '2' ou '5'.")


def desired_poles_from_specs(
    spec_type: str,
    mp_percent: Optional[float] = None,
    ts: Optional[float] = None,
    ts_criterion: str = "2",
    zeta: Optional[float] = None,
    wn: Optional[float] = None,
    explicit_poles: Optional[Sequence[complex]] = None,
) -> Tuple[complex, complex, float, float]:
    spec_type = spec_type.lower()
    if spec_type == "mp_ts":
        if mp_percent is None or ts is None:
            raise ValueError("Informe Mp(%) e t_s para spec_type 'mp_ts'.")
        zeta_val = damping_from_percent_overshoot(mp_percent)
        wn_val = wn_from_settling_time(ts, zeta_val, ts_criterion)
    elif spec_type == "zeta_wn":
        if zeta is None or wn is None:
            raise ValueError("Informe zeta e wn para spec_type 'zeta_wn'.")
        zeta_val = float(zeta)
        wn_val = float(wn)
    elif spec_type == "poles":
        if not explicit_poles:
            raise ValueError("Forneca polos desejados para spec_type 'poles'.")
        candidates = [complex(p) for p in explicit_poles]
        sd = next((c for c in candidates if c.imag > 0), candidates[0])
        zeta_val, wn_val = zeta_wn_from_pole(sd)
        return sd, sd.conjugate(), zeta_val, wn_val
    else:
        raise ValueError("spec_type deve ser 'mp_ts', 'zeta_wn' ou 'poles'.")
    sd = complex(-zeta_val * wn_val, wn_val * math.sqrt(max(0.0, 1.0 - zeta_val**2)))
    return sd, sd.conjugate(), zeta_val, wn_val


def zeta_wn_from_pole(p: complex) -> Tuple[float, float]:
    sigma = -p.real
    wd = abs(p.imag)
    wn = math.hypot(sigma, wd)
    if wn == 0:
        return 1.0, 0.0
    zeta = sigma / wn
    return zeta, wn


def normalize_angle_deg(angle: float) -> float:
    a = (angle + 180.0) % 360.0 - 180.0
    if a == -180.0:
        return 180.0
    return a


def angle_deficit(sd: complex, zeros: Iterable[complex], poles: Iterable[complex]) -> Tuple[float, float]:
    contrib = 0.0
    for z in zeros:
        contrib += angle_deg(sd - z)
    for p in poles:
        contrib -= angle_deg(sd - p)
    angle_at_sd = normalize_angle_deg(contrib)
    raw_def = 180.0 - angle_at_sd
    raw_def = normalize_angle_deg(raw_def)
    acute = abs(raw_def)
    if acute > 180.0:
        acute = 360.0 - acute
    return raw_def, acute


def angle_contributions(
    sd: complex,
    zeros: Iterable[complex],
    poles: Iterable[complex],
) -> Tuple[List[Tuple[complex, float]], List[Tuple[complex, float]], float, float]:
    zero_terms: List[Tuple[complex, float]] = []
    pole_terms: List[Tuple[complex, float]] = []
    total = 0.0
    for z in zeros:
        ang = angle_deg(sd - z)
        zero_terms.append((z, ang))
        total += ang
    for p in poles:
        ang = angle_deg(sd - p)
        pole_terms.append((p, ang))
        total -= ang
    return zero_terms, pole_terms, total, normalize_angle_deg(total)


def zero_location_from_deficit(sd: complex, phi_deg: float) -> float:
    sigma = -sd.real
    wd = abs(sd.imag)
    phi_rad = math.radians(phi_deg)
    if abs(math.tan(phi_rad)) < 1e-12:
        raise ValueError("Angulo muito pequeno para calcular zero.")
    return wd / math.tan(phi_rad) + sigma


def gain_from_magnitude(
    sd: complex,
    plant_zeros: Iterable[complex],
    plant_poles: Iterable[complex],
    ctrl_zeros: Iterable[complex],
    ctrl_poles: Iterable[complex],
    k_plant: float,
    k_sensor: float,
) -> float:
    num = abs(k_plant * k_sensor)
    for z in plant_zeros:
        num *= abs(sd - z)
    for z in ctrl_zeros:
        num *= abs(sd - z)
    den = 1.0
    for p in plant_poles:
        den *= abs(sd - p)
    for p in ctrl_poles:
        den *= abs(sd - p)
    if den == 0:
        raise ValueError("Distancia zero encontrada na condicao de modulo.")
    return 1.0 / (num / den)


# ===== Discretizacao =====
def _rational_poly_eval(coeffs: np.ndarray, s_num: np.ndarray, s_den: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Avalia um polinomio em s substituindo s -> s_num/s_den.

    Retorna (num_z, den_z) tais que resultado = num_z(z) / den_z(z).
    """

    num = np.array([0.0])
    den = np.array([1.0])
    for coef in coeffs:
        num = np.polymul(num, s_num)
        den = np.polymul(den, s_den)
        if abs(coef) > 0:
            num = np.polyadd(num, np.polymul(den, np.array([float(coef)])))
    return num, den


def _substitution_pair(method: str, T: float) -> Tuple[np.ndarray, np.ndarray]:
    method = method.lower()
    if method == "tustin":
        scale = 2.0 / T
        return np.array([scale, -scale]), np.array([1.0, 1.0])
    if method == "euler_forward":
        scale = 1.0 / T
        return np.array([scale, -scale]), np.array([1.0])
    if method == "euler_backward":
        scale = 1.0 / T
        return np.array([scale, -scale]), np.array([1.0, 0.0])
    raise ValueError("Metodo deve ser 'tustin', 'euler_forward' ou 'euler_backward'.")


def _cancel_common_factors(num: np.ndarray, den: np.ndarray, tol: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    num_poly = np.poly1d(num)
    den_poly = np.poly1d(den)

    changed = True
    while changed and num_poly.order >= 0 and den_poly.order >= 0:
        changed = False
        num_roots = np.roots(num_poly) if num_poly.order > 0 else np.array([])
        den_roots = np.roots(den_poly) if den_poly.order > 0 else np.array([])
        for nr in num_roots:
            for dr in den_roots:
                if abs(nr - dr) < tol:
                    num_poly = np.poly1d(np.polydiv(num_poly, np.poly1d([1.0, -nr]))[0])
                    den_poly = np.poly1d(np.polydiv(den_poly, np.poly1d([1.0, -dr]))[0])
                    changed = True
                    break
            if changed:
                break

    num_coeffs = np.real(np.real_if_close(num_poly.coeffs, tol=1000)).astype(float)
    den_coeffs = np.real(np.real_if_close(den_poly.coeffs, tol=1000)).astype(float)
    return num_coeffs, den_coeffs


def discrete_from_continuous(
    num: np.ndarray,
    den: np.ndarray,
    T: float,
    method: str,
    return_steps: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, Dict[str, Union[np.ndarray, float]]],
]:
    """Realiza a substituicao s -> f(z) e retorna o controlador discreto.

    Quando *return_steps* for True, devolve tambem um dicionario com os
    artefatos intermediarios utilizados na obtencao dos coeficientes.
    """

    s_num, s_den = _substitution_pair(method, T)
    num_z_num, num_z_den = _rational_poly_eval(num, s_num, s_den)
    den_z_num, den_z_den = _rational_poly_eval(den, s_num, s_den)

    Nc_poly = np.polymul(num_z_num, den_z_den)
    Dc_poly = np.polymul(num_z_den, den_z_num)

    Nc_reduced, Dc_reduced = _cancel_common_factors(Nc_poly, Dc_poly)

    Nc_trimmed = np.trim_zeros(Nc_reduced, "f")
    Dc_trimmed = np.trim_zeros(Dc_reduced, "f")
    if Dc_trimmed.size == 0:
        raise ValueError("Denominador discreto nulo apos substituicao.")

    leading = Dc_trimmed[0]
    Nc_norm = Nc_trimmed / leading
    Dc_norm = Dc_trimmed / leading

    if not return_steps:
        return Nc_norm, Dc_norm

    steps: Dict[str, Union[np.ndarray, float]] = {
        "s_num": s_num,
        "s_den": s_den,
        "num_z_num": num_z_num,
        "num_z_den": num_z_den,
        "den_z_num": den_z_num,
        "den_z_den": den_z_den,
        "Nc_poly": Nc_poly,
        "Dc_poly": Dc_poly,
        "Nc_reduced": Nc_reduced,
        "Dc_reduced": Dc_reduced,
        "Nc_trimmed": Nc_trimmed,
        "Dc_trimmed": Dc_trimmed,
        "leading": leading,
        "Nc_norm": Nc_norm,
        "Dc_norm": Dc_norm,
    }
    return Nc_norm, Dc_norm, steps

def difference_equation_latex(num: np.ndarray, den: np.ndarray) -> str:
    """Gera um bloco LaTeX (align) para a equação de diferenças padrão."""
    b = num
    a = den / den[0]
    terms_lhs = ["u[k]"]
    for i, coef in enumerate(-a[1:], start=1):
        if abs(coef) > 1e-12:
            formatted = format_signed_number(float(coef))
            if formatted == "0":
                continue
            terms_lhs.append(f"{formatted}\\,u[k-{i}]")
    terms_rhs: List[str] = []
    for i, coef in enumerate(b):
        if abs(coef) > 1e-12:
            if i == 0:
                formatted = format_signed_number(float(coef))
                if formatted == "0":
                    continue
                terms_rhs.append(f"{formatted}\\,e[k]")
            else:
                formatted = format_signed_number(float(coef))
                if formatted == "0":
                    continue
                terms_rhs.append(f"{formatted}\\,e[k-{i}]")
    lhs = " ".join(terms_lhs)
    rhs = " ".join(terms_rhs) if terms_rhs else r"0"
    return lhs + " = " + rhs


def print_difference_equation(rep: Reporter, num: np.ndarray, den: np.ndarray) -> None:
    if rep.mode == "latex":
        rep.text(r"\textbf{Equacao de diferencas (forma padrao)}")
        rep.block(difference_equation_latex(num, den))
    else:
        b = num
        a = den / den[0]
        a_tail = -a[1:]
        a_tail = np.where(np.abs(a_tail) < 1e-9, 0.0, a_tail)
        lhs_terms: List[str] = []
        for idx, coef in enumerate(a_tail, start=1):
            if abs(coef) < 1e-12:
                continue
            formatted = format_signed_number(float(coef))
            if formatted == "0":
                continue
            lhs_terms.append(f"{formatted}*u[k-{idx}]")
        rhs_terms: List[str] = []
        for idx, coef in enumerate(b):
            if abs(coef) < 1e-12:
                continue
            formatted = format_signed_number(float(coef))
            if formatted == "0":
                continue
            rhs_terms.append(f"{formatted}*e[k-{idx}]")
        lhs_expr = "u[k]"
        for term in lhs_terms:
            lhs_expr += f" {term}"
        rhs_expr = " ".join(rhs_terms) if rhs_terms else "0"
        rep.text("   Equacao de diferencas (forma padrao):")
        rep.text(f"      {lhs_expr} = {rhs_expr}")


# ===== Construcao do controlador =====
@dataclass
class ControllerDesignResult:
    plant: TransferFunction
    sensor: TransferFunction
    plant_total: TransferFunction
    controller: TransferFunction
    open_loop: TransferFunction
    closed_loop_poles: np.ndarray
    desired_poles: Tuple[complex, complex]
    gains: Tuple[Optional[float], Optional[float], Optional[float]]
    controller_zeros: List[complex]
    controller_poles: List[complex]
    gain_scalar: float
    report: Optional[str] = None


def design_controller(
    plant: TransferFunction,
    sensor: TransferFunction,
    controller_type: str,
    spec_type: str,
    mp_percent: Optional[float] = None,
    ts: Optional[float] = None,
    ts_criterion: str = "2",
    zeta: Optional[float] = None,
    wn: Optional[float] = None,
    explicit_poles: Optional[Sequence[complex]] = None,
    pid_equal_zeros: bool = True,
    pid_zero_split_deg: Optional[Tuple[float, float]] = None,
    output_mode: str = "text",
    report_file: Optional[str] = None,
    finalize_document: bool = True,
) -> ControllerDesignResult:
    rep = Reporter(mode=output_mode, report_file=report_file, finalize_document=finalize_document)

    controller_type = controller_type.lower()
    sd, sd_conj, zeta_val, wn_val = desired_poles_from_specs(
        spec_type,
        mp_percent=mp_percent,
        ts=ts,
        ts_criterion=ts_criterion,
        zeta=zeta,
        wn=wn,
        explicit_poles=explicit_poles,
    )

    sigma = -sd.real
    wd = abs(sd.imag)

    rep.title("1) Especificacoes \u2192 polos dominantes")
    if spec_type == "mp_ts":
        rep.block(
            latex_align(
                fr"M_p = {format_number(float(mp_percent))}\,\%",
                fr"t_s = {format_number(float(ts))}\,\text{{s}}",
                r"\zeta = -\dfrac{\ln(M_p/100)}{\sqrt{\pi^2 + \ln^2(M_p/100)}}",
                fr"\Rightarrow \zeta = {format_number(zeta_val)}"
            )
        )
        wn_formula = r"\omega_n = \dfrac{3}{\zeta\, t_s}" if ts_criterion == "5" else r"\omega_n = \dfrac{4}{\zeta\, t_s}"
        rep.block(
            latex_align(
                wn_formula,
                fr"\Rightarrow \omega_n = {format_number(wn_val)}\,\text{{rad/s}}"
            )
        )
    elif spec_type == "zeta_wn":
        rep.block(
            latex_align(
                fr"\zeta = {format_number(zeta_val)}",
                fr"\omega_n = {format_number(wn_val)}\,\text{{rad/s}}"
            )
        )
    else:
        rep.text("Polos dominantes informados diretamente.")
        rep.block(
            latex_align(
                fr"\zeta = {format_number(zeta_val)}",
                fr"\omega_n = {format_number(wn_val)}\,\text{{rad/s}}"
            )
        )
    rep.block(
        latex_align(
            fr"\sigma = \zeta\,\omega_n = {format_number(sigma)}",
            fr"\omega_d = \omega_n\sqrt{{1-\zeta^2}} = {format_number(wd)}"
        )
    )
    rep.block(
        latex_align(
            fr"s_d = {format_complex_latex(sd)}",
            fr"s_d^* = {format_complex_latex(sd_conj)}"
        )
    )

    rep.text(r"\textbf{Funcoes de transferencia informadas}")
    plant.pprint(rep, "G")
    sensor.pprint(rep, "H")
    plant_total = plant.series(sensor)
    plant_total.pprint(rep, "L_0")
    plant_zeros = list(plant_total.zeros)
    plant_poles = list(plant_total.poles)
    describe_controller_structure(rep, controller_type, pid_equal_zeros, pid_zero_split_deg)
    detail_open_loop_development(rep, controller_type, plant_total, plant_zeros, plant_poles)

    ctrl_zeros: List[complex] = []
    ctrl_poles: List[complex] = []

    if controller_type in ("pi", "pid"):
        ctrl_poles.append(0.0)

    base_poles = plant_poles + ctrl_poles
    zero_terms, pole_terms, angle_sum_raw, angle_sum_norm = angle_contributions(sd, plant_zeros, base_poles)
    raw_def, acute_def = angle_deficit(sd, plant_zeros, base_poles)

    rep.title("2) Condicao de angulo \u2192 ajuste de zeros/polos do controlador")
    rep.text(fr"Avaliando $L(s_d)$ em $s_d = {format_complex_latex(sd)}$")
    if zero_terms:
        for idx, (z_term, ang_val) in enumerate(zero_terms, start=1):
            rep.block(
                latex_align(
                    fr"+\,\angle(s_d - z_{{{idx}}}) = {format_number(ang_val)}^\circ",
                    fr"\quad \text{{com }} z_{{{idx}}} = {format_complex_latex(z_term)}"
                )
            )
    else:
        rep.text("Sem zeros em L(s).")
    if pole_terms:
        for idx, (p_term, ang_val) in enumerate(pole_terms, start=1):
            rep.block(
                latex_align(
                    fr"-\,\angle(s_d - p_{{{idx}}}) = {format_number(ang_val)}^\circ",
                    fr"\quad \text{{com }} p_{{{idx}}} = {format_complex_latex(p_term)}"
                )
            )
    else:
        rep.text("Sem polos em L(s).")
    rep.block(
        latex_align(
            fr"\text{{Soma bruta}} = {format_number(angle_sum_raw)}^\circ \Rightarrow \text{{normalizada}} = {format_number(angle_sum_norm)}^\circ",
            fr"\text{{Deficit}} = {format_number(raw_def)}^\circ,\quad \text{{agudo}} = {format_number(acute_def)}^\circ"
        )
    )

    if controller_type in ("pi", "pd"):
        z_pos = zero_location_from_deficit(sd, acute_def)
        ctrl_zeros.append(-z_pos)
        rep.block(
            latex_align(
                fr"\phi = {format_number(acute_def)}^\circ",
                r"\tan(\phi) = \dfrac{\omega_d}{x - \sigma}",
                fr"x = \sigma + \dfrac{{\omega_d}}{{\tan(\phi)}} = {format_number(z_pos)}",
                fr"z_c = -x = {format_complex_latex(-z_pos)}"
            )
        )
    elif controller_type == "pid":
        rep.text(r"Polo integrador em $s=0$ considerado.")
        if pid_equal_zeros:
            phi_each = acute_def / 2.0
            z_val = zero_location_from_deficit(sd, phi_each)
            ctrl_zeros.extend([-z_val, -z_val])
            rep.block(
                latex_align(
                    fr"\phi = {format_number(acute_def)}^\circ",
                    r"\phi_1 = \phi_2 = \dfrac{\phi}{2}",
                    fr"x = \sigma + \dfrac{{\omega_d}}{{\tan(\phi/2)}} = {format_number(z_val)}",
                    fr"z_c = -x = {format_complex_latex(-z_val)}\;\text{{(duplicidade)}}"
                )
            )
        else:
            if not pid_zero_split_deg:
                raise ValueError("Informe pid_zero_split_deg para zeros distintos.")
            phi1, phi2 = pid_zero_split_deg
            if abs(phi1 + phi2 - acute_def) > 1e-3:
                phi2 = acute_def - phi1
            z1 = zero_location_from_deficit(sd, phi1)
            z2 = zero_location_from_deficit(sd, phi2)
            ctrl_zeros.extend([-z1, -z2])
            rep.block(
                latex_align(
                    fr"\phi_1 = {format_number(phi1)}^\circ,\quad \phi_2 = {format_number(phi2)}^\circ",
                    r"x_i = \sigma + \dfrac{\omega_d}{\tan\phi_i},\; i=1,2",
                    fr"x_1 = {format_number(sigma)} + \dfrac{{\omega_d}}{{\tan({format_number(phi1)}^\circ)}} = {format_number(z1)}",
                    fr"x_2 = {format_number(sigma)} + \dfrac{{\omega_d}}{{\tan({format_number(phi2)}^\circ)}} = {format_number(z2)}",
                    fr"z_1 = -x_1 = {format_complex_latex(-z1)},\quad z_2 = -x_2 = {format_complex_latex(-z2)}"
                )
            )
    else:
        raise ValueError("controller_type deve ser 'pi', 'pd' ou 'pid'.")

    final_zero_terms, final_pole_terms, final_angle_raw, final_angle_norm = angle_contributions(
        sd, plant_zeros + ctrl_zeros, plant_poles + ctrl_poles
    )
    rep.title("2.1) Verificação da condição de ângulo")
    rep.block(
        latex_align(
            fr"\angle L(s_d)=\sum\theta_z-\sum\theta_p",
            fr"\angle L(s_d)={format_number(final_angle_raw)}^\circ",
            fr"\angle L(s_d)_{{norm}}={format_number(final_angle_norm)}^\circ"
        )
    )

    magnitude_num = abs(plant.gain * sensor.gain)
    numerator_distances: List[Tuple[complex, float]] = []
    for z_term in plant_zeros:
        dist = abs(sd - z_term)
        numerator_distances.append((z_term, dist))
        magnitude_num *= dist
    for z_term in ctrl_zeros:
        dist = abs(sd - z_term)
        numerator_distances.append((z_term, dist))
        magnitude_num *= dist

    magnitude_den = 1.0
    denominator_distances: List[Tuple[complex, float]] = []
    for p_term in plant_poles:
        dist = abs(sd - p_term)
        denominator_distances.append((p_term, dist))
        magnitude_den *= dist
    for p_term in ctrl_poles:
        dist = abs(sd - p_term)
        denominator_distances.append((p_term, dist))
        magnitude_den *= dist

    if magnitude_den == 0:
        raise ValueError("Distancia zero encontrada na condicao de modulo.")

    mag_ratio = magnitude_num / magnitude_den
    k_ctrl = 1.0 / mag_ratio

    rep.title("3) Condicao de modulo \u2192 ganho do controlador")
    rep.eq(r"|K|\prod |s_d - z_i| = \prod |s_d - p_i|")
    rep.block(
        latex_align(
            fr"\lvert G(0)H(0)\rvert = {format_number(abs(plant.gain * sensor.gain))}",
            fr"\prod |s_d - z_i| = {format_number(magnitude_num)}",
            fr"\prod |s_d - p_i| = {format_number(magnitude_den)}"
        )
    )
    if numerator_distances:
        for idx, (z_term, dist) in enumerate(numerator_distances, start=1):
            rep.block(
                latex_align(
                    fr"|s_d - z_{{{idx}}}| = {format_number(dist)}",
                    fr"\quad z_{{{idx}}} = {format_complex_latex(z_term)}"
                )
            )
    if denominator_distances:
        for idx, (p_term, dist) in enumerate(denominator_distances, start=1):
            rep.block(
                latex_align(
                    fr"|s_d - p_{{{idx}}}| = {format_number(dist)}",
                    fr"\quad p_{{{idx}}} = {format_complex_latex(p_term)}"
                )
            )
    rep.block(
        latex_align(
            fr"|L(s_d)| = {format_number(mag_ratio)}\,|K| = 1",
            fr"K = \dfrac{{1}}{{{format_number(mag_ratio)}}} = {format_number(k_ctrl)}"
        )
    )

    ctrl_num = np.array([k_ctrl])
    for zc in ctrl_zeros:
        ctrl_num = np.polymul(ctrl_num, np.array([1.0, -zc]))
    ctrl_den = np.array([1.0])
    for pc in ctrl_poles:
        ctrl_den = np.polymul(ctrl_den, np.array([1.0, -pc]))

    controller_tf = TransferFunction(ctrl_num, ctrl_den)

    rep.title("4) Controlador continuo obtido")
    rep.block(
        latex_align(
            fr"Z(G_c) = {format_complex_list_latex(ctrl_zeros)}",
            fr"P(G_c) = {format_complex_list_latex(ctrl_poles)}"
        )
    )
    controller_tf.pprint(rep, "G_c")

    open_loop_tf = controller_tf.series(plant_total)
    char_poly = poly_add(open_loop_tf.den, open_loop_tf.num)
    closed_poles = np.roots(char_poly)

    rep.title("5) Verificacao \u2192 polos de malha fechada")
    char_poly_expr = poly_to_latex(char_poly)
    poles_expr = ", ".join(format_complex_latex(cp) for cp in closed_poles)
    rep.block(
        latex_align(
            fr"\Delta(s) = {char_poly_expr}",
            fr"\text{{Polos obtidos: }} \{{ {poles_expr} \}}"
        )
    )

    kp = ki = kd = None
    if controller_type == "pd":
        kd = k_ctrl
        kp = k_ctrl * (-ctrl_zeros[0])
    elif controller_type == "pi":
        kp = k_ctrl
        ki = k_ctrl * (-ctrl_zeros[0])
    elif controller_type == "pid":
        kd = k_ctrl
        z1, z2 = -ctrl_zeros[0], -ctrl_zeros[1]
        kp = k_ctrl * (z1 + z2)
        ki = k_ctrl * (z1 * z2)

    rep.title("6) Ganhos equivalentes (forma PID)")
    rep.block(
        latex_align(
            fr"K_p = {format_gain_value(kp)}",
            fr"K_i = {format_gain_value(ki)}",
            fr"K_d = {format_gain_value(kd)}"
        )
    )

    rep.flush()

    return ControllerDesignResult(
        plant=plant,
        sensor=sensor,
        plant_total=plant_total,
        controller=controller_tf,
        open_loop=open_loop_tf,
        closed_loop_poles=closed_poles,
        desired_poles=(sd, sd_conj),
        gains=(kp, ki, kd),
        controller_zeros=ctrl_zeros,
        controller_poles=ctrl_poles,
        gain_scalar=k_ctrl,
        report=rep.dumps(),
    )

def discretize_controller(
    result: ControllerDesignResult,
    T_s: float,
    method: str,
    target: str = "controller",
    output_mode: str = "text",
    report_file: Optional[str] = None,
    continue_document: bool = False,
    finalize_document: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str]:
    rep = Reporter(
        mode=output_mode,
        report_file=report_file,
        continue_document=continue_document,
        finalize_document=finalize_document,
    )

    target_key = target.lower()
    if target_key in {"controller", "gc", "controlador", "controller_only"}:
        target_tf = result.controller
        target_label = "controlador"
        target_symbol = "G_c"
    elif target_key in {"open_loop", "loop", "full", "system", "full_system"}:
        target_tf = result.open_loop
        target_label = "malha aberta completa"
        target_symbol = "L"
    else:
        raise ValueError("target deve ser 'controller' ou 'open_loop'.")

    method_label = {
        "tustin": "Tustin (bilinear)",
        "euler_forward": "Euler para frente",
        "euler_backward": "Euler para tras",
    }.get(method.lower(), method)

    rep.title(f"7) Discretizacao do {target_label}")
    rep.text(fr"Metodo: {method_label}")
    if rep.mode == "latex":
        rep.text(rf"Elemento discretizado: ${target_symbol}(s)$")
    else:
        rep.text(f"Elemento discretizado: {target_symbol}(s)")
    substitution = {
        "tustin": r"s \rightarrow \dfrac{2}{T_s}\,\dfrac{z-1}{z+1}",
        "euler_forward": r"s \rightarrow \dfrac{z-1}{T_s}",
        "euler_backward": r"s \rightarrow \dfrac{1}{T_s}\,\dfrac{z-1}{z}",
    }.get(method.lower(), "s -> substituicao nao listada")
    rep.block(
        latex_align(
            fr"\text{{Substituicao: }} {substitution}",
            fr"T_s = {format_number(T_s)}\,\text{{s}}"
        )
    )
    target_tf.pprint(rep, target_symbol)

    disc_tuple = discrete_from_continuous(
        target_tf.num,
        target_tf.den,
        T_s,
        method,
        return_steps=True,
    )
    Nc, Dc, steps = disc_tuple
    assert steps is not None  # satisfy type checkers; guaranteed by return_steps=True

    rep.text(r"\textbf{Substituicao numerica}")
    s_num = np.array(steps["s_num"], dtype=float)
    s_den = np.array(steps["s_den"], dtype=float)
    rep.block(
        latex_align(
            fr"s_{{\text{{num}}}}(z) = {poly_to_latex(s_num, 'z')}",
            fr"s_{{\text{{den}}}}(z) = {poly_to_latex(s_den, 'z')}"
        )
    )

    num_z_num = np.array(steps["num_z_num"], dtype=float)
    num_z_den = np.array(steps["num_z_den"], dtype=float)
    den_z_num = np.array(steps["den_z_num"], dtype=float)
    den_z_den = np.array(steps["den_z_den"], dtype=float)
    rep.text(r"\textbf{Resultado apos substituir em } $G_c(s)$")
    rep.block(
        latex_align(
            fr"N_s(z) = {poly_to_latex(num_z_num, 'z')},\quad D_s(z) = {poly_to_latex(num_z_den, 'z')}",
            fr"N_p(z) = {poly_to_latex(den_z_num, 'z')},\quad D_p(z) = {poly_to_latex(den_z_den, 'z')}"
        )
    )

    Nc_raw = np.array(steps["Nc_poly"], dtype=float)
    Dc_raw = np.array(steps["Dc_poly"], dtype=float)
    rep.text(r"\textbf{Combinação em denominador comum}")
    rep.block(
        latex_align(
            "N_c^{\\text{raw}}(z) = N_s(z)D_p(z) = " + poly_to_latex(Nc_raw, "z"),
            "D_c^{\\text{raw}}(z) = D_s(z)N_p(z) = " + poly_to_latex(Dc_raw, "z")
        )
    )

    Nc_reduced = np.array(steps["Nc_reduced"], dtype=float)
    Dc_reduced = np.array(steps["Dc_reduced"], dtype=float)
    same_nc = Nc_raw.shape == Nc_reduced.shape and np.allclose(Nc_raw, Nc_reduced)
    same_dc = Dc_raw.shape == Dc_reduced.shape and np.allclose(Dc_raw, Dc_reduced)
    if same_nc and same_dc:
        rep.text("Nao ha fatores comuns a cancelar.")
    else:
        rep.text(r"\textbf{Cancelamento de fatores comuns}")
        rep.block(
            latex_align(
                "N_c^{\\text{reduc}}(z) = " + poly_to_latex(Nc_reduced, "z"),
                "D_c^{\\text{reduc}}(z) = " + poly_to_latex(Dc_reduced, "z")
            )
        )

    Nc_trimmed = np.array(steps["Nc_trimmed"], dtype=float)
    Dc_trimmed = np.array(steps["Dc_trimmed"], dtype=float)
    rep.text(r"\textbf{Remocao de coeficientes nulos a esquerda}")
    rep.block(
        latex_align(
            "N_c^{\\text{trim}}(z) = " + poly_to_latex(Nc_trimmed, "z"),
            "D_c^{\\text{trim}}(z) = " + poly_to_latex(Dc_trimmed, "z")
        )
    )

    leading = float(steps["leading"])
    rep.text(r"\textbf{Normalizacao para denominador monico}")
    inv_leading = 1.0 / leading
    rep.block(
        latex_align(
            f"a_0 = {format_number(leading)}",
            f"N_c(z) = {format_number(inv_leading)} \\cdot N_c^{{\\text{{trim}}}}(z) = {poly_to_latex(Nc, 'z')}",
            f"D_c(z) = {format_number(inv_leading)} \\cdot D_c^{{\\text{{trim}}}}(z) = {poly_to_latex(Dc, 'z')}"
        )
    )

    rep.eq(fr"{target_symbol}(z) = \dfrac{{{poly_to_latex(Nc, 'z')}}}{{{poly_to_latex(Dc, 'z')}}}")
    rep.text(r"\textbf{Coeficientes (forma direta)}")
    num_lines = ",\\; ".join(
        fr"b_{idx} = {format_number(float(coef))}" for idx, coef in enumerate(Nc)
    )
    den_lines = ",\\; ".join(
        fr"a_{idx} = {format_number(float(coef))}" for idx, coef in enumerate(Dc)
    )
    rep.block(latex_align(fr"{num_lines}", fr"{den_lines}"))

    rep.text(r"\textbf{Polinomios apos substituicao e normalizacao}")
    rep.block(
        latex_align(
            fr"\text{{Numerador: }} {format_array_latex(Nc)}",
            fr"\text{{Denominador: }} {format_array_latex(Dc)}"
        )
    )
    print_difference_equation(rep, Nc, Dc)

    rep.flush()
    return Nc, Dc, rep.dumps()

# INTERFACE STREAMLIT — DESENVOLVIMENTO PASSO A PASSO
import streamlit as st
import matplotlib.pyplot as plt
import re

st.set_page_config(
    page_title="Projeto de Controladores PI / PD / PID",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def parse_coefficients(text: str):
    text = text.strip()
    if not text:
        raise ValueError("A lista de coeficientes não pode estar vazia.")
    try:
        values = [float(x) for x in re.split(r"[,;\s]+", text) if x]
    except ValueError as exc:
        raise ValueError("Use números separados por vírgula, ponto e vírgula ou espaço.") from exc
    if not values:
        raise ValueError("Nenhum coeficiente foi informado.")
    return np.asarray(values, dtype=float)


def complex_to_text(z):
    return format_complex_str(complex(z))


def tf_text(tf):
    return f"({tf._poly_to_string(tf.num, 's')}) / ({tf._poly_to_string(tf.den, 's')})"


def tf_latex(tf, variable="s"):
    return rf"\frac{{{poly_to_latex(tf.num, variable)}}}{{{poly_to_latex(tf.den, variable)}}}"


def show_complex_list(values, label=None):
    if label:
        st.markdown(f"**{label}**")
    values = list(values)
    if not values:
        st.write("Nenhum.")
        return
    for i, value in enumerate(values, 1):
        st.latex(rf"p_{i} = {format_complex_latex(complex(value))}")


def plot_pole_zero_map(result, plant_total, desired_poles):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)

    zeros = np.asarray(plant_total.zeros, dtype=complex)
    poles = np.asarray(plant_total.poles, dtype=complex)
    ctrl_zeros = np.asarray(result.controller_zeros, dtype=complex)
    ctrl_poles = np.asarray(result.controller_poles, dtype=complex)
    closed = np.asarray(result.closed_loop_poles, dtype=complex)
    desired = np.asarray(desired_poles, dtype=complex)

    if len(zeros):
        ax.scatter(zeros.real, zeros.imag, marker="o", s=90,
                   facecolors="none", label="Zeros G(s)H(s)")
    if len(poles):
        ax.scatter(poles.real, poles.imag, marker="x", s=90,
                   label="Polos G(s)H(s)")
    if len(ctrl_zeros):
        ax.scatter(ctrl_zeros.real, ctrl_zeros.imag, marker="o", s=90,
                   facecolors="none", label="Zeros do controlador")
    if len(ctrl_poles):
        ax.scatter(ctrl_poles.real, ctrl_poles.imag, marker="x", s=90,
                   label="Polos do controlador")
    if len(desired):
        ax.scatter(desired.real, desired.imag, marker="*", s=160,
                   label="Polos desejados")
    if len(closed):
        ax.scatter(closed.real, closed.imag, marker="+", s=110,
                   label="Polos malha fechada")

    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Parte real")
    ax.set_ylabel("Parte imaginária")
    ax.set_title("Mapa de polos e zeros")
    ax.legend()
    fig.tight_layout()
    return fig


def render_design_latex(result, G, H, controller_type, spec_data):
    """Mostra o desenvolvimento contínuo completo, passo a passo."""
    sd = complex(result.desired_poles[0])
    sd_conj = complex(result.desired_poles[1])
    zeta_val, wn_val = zeta_wn_from_pole(sd)
    sigma = -sd.real
    wd = abs(sd.imag)

    plant_total = result.plant_total
    plant_zeros = list(plant_total.zeros)
    plant_poles = list(plant_total.poles)
    ctrl_poles = list(result.controller_poles)
    ctrl_zeros = list(result.controller_zeros)

    # PASSO 1
    st.markdown("## 1. Especificações → polos dominantes")

    if spec_data["mode"] == "Mp_ts":
        mp = spec_data["mp"]
        ts = spec_data["ts"]
        criterion = spec_data["criterion"]
        st.markdown("A partir do sobressinal e do tempo de acomodação:")
        st.latex(
            rf"M_p = {format_number(mp)}\%,\qquad "
            rf"t_s = {format_number(ts)}\,s"
        )
        st.latex(
            r"\zeta = -\frac{\ln(M_p/100)}{"
            r"\sqrt{\pi^2+\ln^2(M_p/100)}}"
        )
        st.latex(rf"\therefore\quad \zeta = {format_number(zeta_val)}")
        constant = 3 if str(criterion) == "5" else 4
        st.latex(
            rf"\omega_n = \frac{{{constant}}}{{\zeta t_s}}"
            rf" = \frac{{{constant}}}{{({format_number(zeta_val)})({format_number(ts)})}}"
            rf" = {format_number(wn_val)}\,rad/s"
        )
    elif spec_data["mode"] == "zeta_wn":
        st.latex(
            rf"\zeta = {format_number(zeta_val)},\qquad "
            rf"\omega_n = {format_number(wn_val)}\,rad/s"
        )
    else:
        st.markdown("Os polos dominantes foram informados diretamente:")
        st.latex(
            rf"s_d = {format_complex_latex(sd)},\qquad "
            rf"s_d^* = {format_complex_latex(sd_conj)}"
        )

    st.latex(
        rf"\sigma = \zeta\omega_n = {format_number(sigma)}"
    )
    st.latex(
        rf"\omega_d = \omega_n\sqrt{{1-\zeta^2}} = {format_number(wd)}"
    )
    st.latex(
        rf"\boxed{{s_d = {format_complex_latex(sd)}}},\qquad "
        rf"\boxed{{s_d^* = {format_complex_latex(sd_conj)}}}"
    )

    st.markdown("---")

    st.markdown("## 2. Funções de transferência e estrutura do controlador")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Planta**")
        st.latex(rf"G(s) = {tf_latex(G)}")
    with c2:
        st.markdown("**Sensor**")
        st.latex(rf"H(s) = {tf_latex(H)}")

    st.latex(r"L_0(s)=G(s)H(s)")
    st.latex(rf"L_0(s) = {tf_latex(plant_total)}")

    st.markdown(f"Para um controlador **{controller_type}**:")
    if controller_type == "PD":
        st.latex(r"G_c(s)=K(s-z_c)")
    elif controller_type == "PI":
        st.latex(r"G_c(s)=K\frac{s-z_c}{s}")
    else:
        st.latex(r"G_c(s)=K\frac{(s-z_1)(s-z_2)}{s}")

    st.markdown("---")

    # PASSO 3
    st.markdown("## 3. Condição de ângulo")
    st.markdown("Para que $s_d$ pertença ao lugar das raízes, deve valer:")
    st.latex(r"\angle L(s_d)=(2k+1)180^\circ")

    st.markdown(
        "Primeiro calculamos as contribuições dos elementos conhecidos. "
        "O(s) zero(s) do controlador ainda é(são) incógnita(s)."
    )

    base_poles = plant_poles + ctrl_poles
    zero_terms, pole_terms, angle_sum_raw, angle_sum_norm = angle_contributions(
        sd, plant_zeros, base_poles
    )
    raw_def, required_zero_angle = angle_deficit(
        sd, plant_zeros, base_poles
    )

    if plant_zeros:
        st.markdown("**Contribuição dos zeros da planta/sensor:**")
        for i, (z, ang) in enumerate(zero_terms, 1):
            st.latex(
                rf"\theta_{{z_{i}}}=\angle(s_d-z_{i})"
                rf"=\angle\left({format_complex_latex(sd)}"
                rf"-({format_complex_latex(z)})\right)"
                rf"={format_number(ang)}^\circ"
            )
    else:
        st.write("A planta/sensor não possui zeros finitos.")

    if pole_terms:
        st.markdown("**Contribuição dos polos:**")
        for i, (p, ang) in enumerate(pole_terms, 1):
            st.latex(
                rf"\theta_{{p_{i}}}=\angle(s_d-p_{i})"
                rf"=\angle\left({format_complex_latex(sd)}"
                rf"-({format_complex_latex(p)})\right)"
                rf"={format_number(ang)}^\circ"
            )

    st.markdown("**Soma das contribuições conhecidas:**")
    st.latex(
        rf"\sum\theta_{{conhecidos}}=\sum\theta_z-\sum\theta_p"
        rf"={format_number(angle_sum_raw)}^\circ"
    )
    st.latex(
        rf"\sum\theta_{{conhecidos,norm}}={format_number(angle_sum_norm)}^\circ"
    )

    theta_zc = required_zero_angle

    if controller_type in ("PD", "PI"):
        st.markdown("**Condição de ângulo incluindo o zero do controlador:**")
        st.latex(
            r"\sum\theta_{z,\mathrm{planta}}+\theta_{z_c}-\sum\theta_p"
            r"=(2k+1)180^\circ"
        )
        st.latex(
            rf"\theta_{{z_c}}=180^\circ-\left({format_number(angle_sum_norm)}^\circ\right)"
            rf"={format_number(theta_zc)}^\circ"
        )
        st.latex(rf"\boxed{{|\theta_{{z_c}}|={format_number(abs(theta_zc))}^\circ}}")

    elif controller_type == "PID":
        st.markdown("**Condição de ângulo incluindo os zeros do controlador:**")
        st.latex(
            r"\sum\theta_{z,\mathrm{planta}}+\theta_{z_1}+\theta_{z_2}-\sum\theta_p"
            r"=(2k+1)180^\circ"
        )
        st.latex(
            rf"\theta_{{z_1}}+\theta_{{z_2}}=180^\circ-\left({format_number(angle_sum_norm)}^\circ\right)"
            rf"={format_number(theta_zc)}^\circ"
        )
        st.latex(rf"\boxed{{|\theta_{{z_1}}+\theta_{{z_2}}|={format_number(abs(theta_zc))}^\circ}}")

    st.markdown("### Localização dos zeros do controlador")

    if controller_type in ("PD", "PI") and ctrl_zeros:
        zc = ctrl_zeros[0]
        st.latex(r"\tan(\theta_{z_c})=\frac{\omega_d}{\operatorname{Re}(s_d)-z_c}")
        st.latex(r"z_c=\operatorname{Re}(s_d)-\frac{\omega_d}{\tan(\theta_{z_c})}")
        st.latex(
            rf"z_c={format_number(sd.real)}-\frac{{{format_number(wd)}}}"
            rf"{{\tan({format_number(theta_zc)}^\circ)}}"
            rf"={format_number(zc.real)}"
        )
        st.latex(rf"\boxed{{z_c={format_complex_latex(zc)}}}")

    elif controller_type == "PID" and len(ctrl_zeros) >= 2:
        if abs(ctrl_zeros[0] - ctrl_zeros[1]) < 1e-9:
            phi_each = theta_zc / 2.0
            zc = ctrl_zeros[0]
            st.latex(r"\theta_{z_1}=\theta_{z_2}=\frac{\theta_{z_1}+\theta_{z_2}}{2}")
            st.latex(rf"\theta_{{z_1}}=\theta_{{z_2}}={format_number(phi_each)}^\circ")
            st.latex(
                rf"z_1=z_2={format_number(sd.real)}-\frac{{{format_number(wd)}}}"
                rf"{{\tan({format_number(phi_each)}^\circ)}}={format_number(zc.real)}"
            )
            st.latex(rf"\boxed{{z_1=z_2={format_complex_latex(zc)}}}")
        else:
            for i, zc in enumerate(ctrl_zeros, 1):
                phi_i = math.degrees(math.atan2(wd, sd.real - zc.real))
                st.latex(rf"\theta_{{z_{i}}}={format_number(phi_i)}^\circ")
                st.latex(rf"\boxed{{z_{i}={format_complex_latex(zc)}}}")

    if ctrl_zeros or ctrl_poles:
        final_zeros = plant_zeros + ctrl_zeros
        final_poles = plant_poles + ctrl_poles
        _, _, final_angle_raw, final_angle_norm = angle_contributions(
            sd, final_zeros, final_poles
        )
        st.markdown("### Verificação da condição de ângulo")
        st.latex(r"\angle L(s_d)=\sum\theta_z-\sum\theta_p")
        st.latex(
            rf"\angle L(s_d)={format_number(final_angle_raw)}^\circ"
            rf"\quad\Rightarrow\quad"
            rf"\angle L(s_d)_{{norm}}={format_number(final_angle_norm)}^\circ"
        )
        if abs(abs(final_angle_norm) - 180.0) < 1e-3:
            st.success("Condição de ângulo satisfeita: o polo desejado pertence ao lugar das raízes.")
        else:
            st.warning("A condição de ângulo não foi satisfeita dentro da tolerância numérica.")

    st.markdown("---")

    # PASSO 4
    st.markdown("## 4. Condição de magnitude → cálculo de K")
    st.markdown("No polo desejado, impomos $|L(s_d)|=1$:")
    st.latex(r"|K|\frac{\prod_i|s_d-z_i|}{\prod_j|s_d-p_j|}=1")

    plant_gain = abs(plant_total.gain)
    numerator_product = plant_gain
    denominator_product = 1.0

    st.latex(rf"|G(s)H(s)|\text{{ (fator de ganho) }} = {format_number(plant_gain)}")

    st.markdown("**Distâncias até os zeros:**")
    for i, z in enumerate(plant_zeros + ctrl_zeros, 1):
        d = abs(sd - z)
        numerator_product *= d
        st.latex(
            rf"|s_d-z_{i}|=|{format_complex_latex(sd)}"
            rf"-({format_complex_latex(z)})|={format_number(d)}"
        )

    st.markdown("**Distâncias até os polos:**")
    for i, p in enumerate(plant_poles + ctrl_poles, 1):
        d = abs(sd - p)
        denominator_product *= d
        st.latex(
            rf"|s_d-p_{i}|=|{format_complex_latex(sd)}"
            rf"-({format_complex_latex(p)})|={format_number(d)}"
        )

    ratio = numerator_product / denominator_product
    st.latex(
        rf"|L(s_d)|=K\left(\frac{{{format_number(numerator_product)}}}"
        rf"{{{format_number(denominator_product)}}}\right)=1"
    )
    st.latex(
        rf"K=\frac{{1}}{{{format_number(ratio)}}}"
        rf"={format_number(result.gain_scalar)}"
    )
    st.latex(rf"\boxed{{K={format_number(result.gain_scalar)}}}")

    st.markdown("---")

    # PASSO 5
    st.markdown("## 5. Controlador contínuo e verificação")
    st.latex(rf"G_c(s)={tf_latex(result.controller)}")

    st.markdown("**Zeros e polos do controlador:**")
    show_complex_list(ctrl_zeros, "Zeros")
    show_complex_list(ctrl_poles, "Polos")

    st.markdown("### Equação característica")
    char_poly = poly_add(result.open_loop.den, result.open_loop.num)
    st.latex(rf"\Delta(s)={poly_to_latex(char_poly)}=0")
    st.markdown("Polos obtidos da malha fechada:")
    show_complex_list(result.closed_loop_poles, "Polos")

    kp, ki, kd = result.gains
    st.markdown("### Ganhos equivalentes")
    if kp is not None:
        st.latex(rf"K_p={format_number(kp)}")
    if ki is not None:
        st.latex(rf"K_i={format_number(ki)}")
    if kd is not None:
        st.latex(rf"K_d={format_number(kd)}")


def render_discretization_steps(result, T_s, method, target):
    """Mostra a discretização como desenvolvimento algébrico."""
    target_key = target.lower()
    if target_key == "controller":
        target_tf = result.controller
        symbol = "G_c"
    else:
        target_tf = result.open_loop
        symbol = "L"

    method_key = method.lower()
    substitution = {
        "tustin": r"s\rightarrow\frac{2}{T_s}\frac{z-1}{z+1}",
        "euler_forward": r"s\rightarrow\frac{z-1}{T_s}",
        "euler_backward": r"s\rightarrow\frac{z-1}{T_s z}",
    }[method_key]

    st.markdown("## 6. Discretização — desenvolvimento")
    st.latex(rf"T_s={format_number(T_s)}\,s")
    st.latex(substitution)
    st.latex(rf"{symbol}(s)={tf_latex(target_tf)}")

    Nc, Dc, steps = discrete_from_continuous(
        target_tf.num, target_tf.den, T_s, method, return_steps=True
    )

    st.markdown("### 6.1 Substituição em cada polinômio")
    s_num = np.asarray(steps["s_num"], dtype=float)
    s_den = np.asarray(steps["s_den"], dtype=float)
    st.latex(rf"s=\frac{{{poly_to_latex(s_num,'z')}}}{{{poly_to_latex(s_den,'z')}}}")

    num_z_num = np.asarray(steps["num_z_num"], dtype=float)
    num_z_den = np.asarray(steps["num_z_den"], dtype=float)
    den_z_num = np.asarray(steps["den_z_num"], dtype=float)
    den_z_den = np.asarray(steps["den_z_den"], dtype=float)

    st.latex(
        rf"N_s(z)={poly_to_latex(num_z_num,'z')},\qquad "
        rf"D_s(z)={poly_to_latex(num_z_den,'z')}"
    )
    st.latex(
        rf"N_p(z)={poly_to_latex(den_z_num,'z')},\qquad "
        rf"D_p(z)={poly_to_latex(den_z_den,'z')}"
    )

    st.markdown("### 6.2 Colocando em denominador comum")
    Nc_raw = np.asarray(steps["Nc_poly"], dtype=float)
    Dc_raw = np.asarray(steps["Dc_poly"], dtype=float)
    st.latex(
        rf"N_c^{{raw}}(z)=N_s(z)D_p(z)={poly_to_latex(Nc_raw,'z')}"
    )
    st.latex(
        rf"D_c^{{raw}}(z)=D_s(z)N_p(z)={poly_to_latex(Dc_raw,'z')}"
    )

    Nc_reduced = np.asarray(steps["Nc_reduced"], dtype=float)
    Dc_reduced = np.asarray(steps["Dc_reduced"], dtype=float)
    if not (np.allclose(Nc_raw, Nc_reduced) and np.allclose(Dc_raw, Dc_reduced)):
        st.markdown("### 6.3 Cancelamento de fatores comuns")
        st.latex(rf"N_c^{{red}}(z)={poly_to_latex(Nc_reduced,'z')}")
        st.latex(rf"D_c^{{red}}(z)={poly_to_latex(Dc_reduced,'z')}")
    else:
        st.markdown("### 6.3 Cancelamento de fatores comuns")
        st.info("Não foram encontrados fatores comuns para cancelar.")

    Nc_trimmed = np.asarray(steps["Nc_trimmed"], dtype=float)
    Dc_trimmed = np.asarray(steps["Dc_trimmed"], dtype=float)
    st.markdown("### 6.4 Remoção de zeros à esquerda")
    st.latex(rf"N_c^{{trim}}(z)={poly_to_latex(Nc_trimmed,'z')}")
    st.latex(rf"D_c^{{trim}}(z)={poly_to_latex(Dc_trimmed,'z')}")

    leading = float(steps["leading"])
    st.markdown("### 6.5 Normalização do denominador")
    st.latex(rf"a_0={format_number(leading)}")
    st.latex(
        rf"N_c(z)=\frac{{N_c^{{trim}}(z)}}{{a_0}}"
        rf"={poly_to_latex(Nc,'z')}"
    )
    st.latex(
        rf"D_c(z)=\frac{{D_c^{{trim}}(z)}}{{a_0}}"
        rf"={poly_to_latex(Dc,'z')}"
    )

    st.markdown("### 6.6 Função de transferência discreta final")
    st.latex(rf"\boxed{{{symbol}(z)=\frac{{{poly_to_latex(Nc,'z')}}}{{{poly_to_latex(Dc,'z')}}}}}")

    st.markdown("### 6.7 Equação de diferenças")
    st.latex(difference_equation_latex(Nc, Dc))

    return Nc, Dc

# SIDEBAR
st.title("Projeto de Controladores PI / PD / PID")
st.caption("Desenvolvimento completo do projeto com fórmulas em LaTeX.")

with st.sidebar:
    st.header("Configuração")

    st.subheader("1. Sistema")
    plant_num_text = st.text_input("Numerador G(s)", value="4, 16")
    plant_den_text = st.text_input("Denominador G(s)", value="1, 4, 4, 0")
    sensor_num_text = st.text_input("Numerador H(s)", value="1")
    sensor_den_text = st.text_input("Denominador H(s)", value="1")

    st.divider()
    st.subheader("2. Controlador")
    controller_type = st.selectbox("Tipo", ["PD", "PI", "PID"])

    st.divider()
    st.subheader("3. Especificações")
    spec_mode = st.selectbox(
        "Definição dos polos",
        ["Mp (%) e tempo de acomodação", "ζ e ωn", "Polos desejados diretamente"],
    )

    mp = ts = zeta = wn = None
    pole_real = pole_imag = None
    settling_criterion = "5"

    if spec_mode == "Mp (%) e tempo de acomodação":
        mp = st.number_input("Mp (%)", min_value=0.01, value=10.0, step=0.5)
        ts = st.number_input("t_s (s)", min_value=0.001, value=4.0, step=0.5)
        settling_criterion = st.selectbox("Critério de t_s", ["2", "5"], index=1)
    elif spec_mode == "ζ e ωn":
        zeta = st.number_input("ζ", min_value=0.001, max_value=0.999,
                               value=0.5912, step=0.01)
        wn = st.number_input("ω_n (rad/s)", min_value=0.001,
                             value=1.5, step=0.1)
    else:
        pole_real = st.number_input("Re{s_d}", value=-1.0, step=0.1)
        pole_imag = st.number_input("Im{s_d}", value=1.0, step=0.1)

    pid_equal_zeros = True
    if controller_type == "PID":
        st.divider()
        pid_equal_zeros = st.radio(
            "Zeros do PID",
            ["Zeros iguais", "Zeros distintos"],
        ) == "Zeros iguais"

    st.divider()
    st.subheader("4. Discretização")
    discretize = st.checkbox("Discretizar", value=True)
    sample_time = 2.0
    discretization_method = "euler_backward"
    discretization_target = "controller"

    if discretize:
        sample_time = st.number_input("T_s (s)", min_value=1e-6, value=2.0, step=0.1)
        method_label = st.selectbox(
            "Método", ["Tustin", "Euler para frente", "Euler para trás"]
        )
        discretization_method = {
            "Tustin": "tustin",
            "Euler para frente": "euler_forward",
            "Euler para trás": "euler_backward",
        }[method_label]
        target_label = st.selectbox("Alvo", ["Controlador Gc(s)", "Malha aberta L(s)"])
        discretization_target = "controller" if target_label.startswith("Controlador") else "open_loop"

    st.divider()
    run = st.button("Projetar controlador", type="primary", use_container_width=True)

# EXECUÇÃO
if run:
    try:
        G = TransferFunction.from_coeffs(
            parse_coefficients(plant_num_text),
            parse_coefficients(plant_den_text),
        )
        H = TransferFunction.from_coeffs(
            parse_coefficients(sensor_num_text),
            parse_coefficients(sensor_den_text),
        )

        if spec_mode == "Mp (%) e tempo de acomodação":
            result = design_controller(
                plant=G, sensor=H,
                controller_type=controller_type.lower(),
                spec_type="mp_ts", mp_percent=mp, ts=ts,
                ts_criterion=settling_criterion,
                pid_equal_zeros=pid_equal_zeros,
                output_mode="text",
            )
        elif spec_mode == "ζ e ωn":
            result = design_controller(
                plant=G, sensor=H,
                controller_type=controller_type.lower(),
                spec_type="zeta_wn", zeta=zeta, wn=wn,
                pid_equal_zeros=pid_equal_zeros,
                output_mode="text",
            )
        else:
            sd = complex(pole_real, pole_imag)
            result = design_controller(
                plant=G, sensor=H,
                controller_type=controller_type.lower(),
                spec_type="poles", explicit_poles=[sd, sd.conjugate()],
                pid_equal_zeros=pid_equal_zeros,
                output_mode="text",
            )

        st.session_state.result = result
        st.session_state.G = G
        st.session_state.H = H
        st.session_state.design_data = {
            "mode": "Mp_ts" if spec_mode == "Mp (%) e tempo de acomodação" else
                    "zeta_wn" if spec_mode == "ζ e ωn" else "poles",
            "mp": mp, "ts": ts, "criterion": settling_criterion,
            "zeta": zeta, "wn": wn,
            "pole_real": pole_real, "pole_imag": pole_imag,
            "controller_type": controller_type,
            "discretize": discretize,
            "sample_time": sample_time,
            "method": discretization_method,
            "target": discretization_target,
        }
        st.session_state.error = None
    except Exception as exc:
        st.session_state.result = None
        st.session_state.error = str(exc)

if st.session_state.get("error"):
    st.error(st.session_state.error)
    st.stop()

result = st.session_state.get("result")

if result is None:
    st.info("Configure o sistema e clique em **Projetar controlador**.")
    st.markdown("### O que será mostrado")
    cols = st.columns(6)
    for col, n, title in zip(
        cols, range(1, 7),
        ["Especificações", "Polos", "Ângulo", "Ganho", "Controlador", "Discreto"]
    ):
        with col:
            st.markdown(f"### {n}. {title}")
    st.markdown("### Exemplo padrão")
    st.latex(r"G(s)=\frac{4s+16}{s^3+4s^2+4s}")
    st.latex(r"H(s)=1")
    st.stop()

G = st.session_state.G
H = st.session_state.H
data = st.session_state.design_data
kp, ki, kd = result.gains

st.success("Projeto concluído.")

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("K", format_number(result.gain_scalar))
with m2:
    st.metric("Kp", "—" if kp is None else format_number(kp))
with m3:
    st.metric("Ki", "—" if ki is None else format_number(ki))
with m4:
    st.metric("Kd", "—" if kd is None else format_number(kd))

# ABAS
tab_dev, tab_map, tab_disc, tab_report = st.tabs(
    ["Desenvolvimento passo a passo", "Polos e zeros", "Discretização", "Relatório"]
)

with tab_dev:
    render_design_latex(
        result, G, H, data["controller_type"],
        {
            "mode": data["mode"],
            "mp": data["mp"],
            "ts": data["ts"],
            "criterion": data["criterion"],
        },
    )

with tab_map:
    desired = np.asarray(result.desired_poles, dtype=complex)
    st.subheader("Mapa completo")
    fig = plot_pole_zero_map(result, result.plant_total, desired)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    c1, c2 = st.columns(2)
    with c1:
        show_complex_list(result.plant_total.poles, "Polos de G(s)H(s)")
        show_complex_list(result.plant_total.zeros, "Zeros de G(s)H(s)")
    with c2:
        show_complex_list(result.controller_poles, "Polos do controlador")
        show_complex_list(result.controller_zeros, "Zeros do controlador")

    st.markdown("### Polos desejados")
    show_complex_list(desired)
    st.markdown("### Polos de malha fechada")
    show_complex_list(result.closed_loop_poles)

with tab_disc:
    if not data["discretize"]:
        st.info("A discretização foi desativada.")
    else:
        render_discretization_steps(
            result,
            data["sample_time"],
            data["method"],
            data["target"],
        )

with tab_report:
    st.subheader("Relatório gerado pelo núcleo matemático")
    st.code(result.report or "", language="text")
    st.markdown("### Função de transferência contínua")
    st.latex(rf"G_c(s)={tf_latex(result.controller)}")
    st.markdown("### Ganhos")
    if kp is not None:
        st.latex(rf"K_p={format_number(kp)}")
    if ki is not None:
        st.latex(rf"K_i={format_number(ki)}")
    if kd is not None:
        st.latex(rf"K_d={format_number(kd)}")
