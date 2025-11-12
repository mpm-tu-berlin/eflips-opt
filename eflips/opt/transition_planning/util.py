def npv_with_escalation(cost, escalation_rate, discount_rate, year):
    """
    Calculate the net present value (NPV) of a cost with escalation over a given number of years.
    :param cost: Initial cost
    :param escalation_rate: Annual escalation rate (as a decimal)
    :param discount_rate: Annual discount rate (as a decimal)
    :param year: Year for which to calculate the NPV
    :return: NPV of the cost in the given year
    """
    escalated_cost = cost * (1 + escalation_rate) ** year
    npv = escalated_cost / ((1 + discount_rate) ** year)
    return npv
