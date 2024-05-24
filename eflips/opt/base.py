from abc import abstractmethod, ABC
from typing import Dict, Any
from sqlalchemy.orm import Session


class AbstractOptimizer(ABC):
    def __init__(self, session: Session, scenario_id: int):
        """
        Constructor for the AbstractOptimizer class. It takes a SQLAlchemy session as an argument, which will grant
        access to the database. The creation and close of this session should be handled outside of this class.
        :param session: A SQLAlchemy session object.
        """
        self.session = session
        self.scenario_id = scenario_id
        self.data = None

    @abstractmethod
    def delete_original_data(self):
        """
        Delete a part of data from the database. These data are the target of this optimization and will be replaced
        with new data.
        """
        pass

    @abstractmethod
    def get_user_inputs(self, user_input: Dict[str, Any]):
        """
        Get the new data for the optimization problem from the user. They will not be stored in the database yet.
        """
        pass

    @abstractmethod
    def data_preparation(self):
        """
        Prepare the data for the optimization problem. This method should be called after get_inputs. The data will
        be stored in self.data
        """
        pass

    @abstractmethod
    def optimize(self):
        """
        Solve the optimization problem using the data stored in self.data.
        """
        pass

    @abstractmethod
    def write_optimization_results(self):
        """
        Update the database with the results of the optimization problem stored in self.data.
        """
        pass

    @abstractmethod
    def visualize(self):
        """
        Visualize the results of the optimization problem stored in self.data.
        """
        pass
