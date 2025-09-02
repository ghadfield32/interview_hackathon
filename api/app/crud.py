from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from .models import User

async def get_user_by_username(db: AsyncSession, username: str):
    stmt = select(User).where(User.username == username)
    res = await db.execute(stmt)
    return res.scalar_one_or_none() 
